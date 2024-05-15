//
// Created by quatinhe on 5/7/24.
//

#include "histogram_eq.h"
#include <chrono>
#include <iostream>
#include <omp.h>
#include <cub/cub.cuh>

#define cudaCheckError() {                                           \
    cudaError_t e = cudaGetLastError();                              \
    if (e != cudaSuccess) {                                          \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
}
namespace cp {

    constexpr auto HISTOGRAM_LENGTH = 256; // Corresponde ao número de tons de cinza em 8bit

    static float inline prob(const int x, const int size) {
        return (float) x / (float) size;
    }

    static unsigned char inline clamp(unsigned char x) { //assegura q um valor está no range
        return std::min(std::max(x, static_cast<unsigned char>(0)), static_cast<unsigned char>(255));
    }

    static unsigned char inline correct_color(float cdf_val, float cdf_min) {
        return clamp(static_cast<unsigned char>(255 * (cdf_val - cdf_min) / (1 - cdf_min)));
    }


    void convertToGrayscale(const unsigned char* input_image, unsigned char* output_image, int width, int height) {
        /** O Colapse neste contexto permite o OpenMP distribuir as iterações de ambos os loops pelas threads*/
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                int idx = i * width + j;
                // Access RGB values from the input image
                auto r = input_image[3 * idx];     // Red value
                auto g = input_image[3 * idx + 1]; // Green value
                auto b = input_image[3 * idx + 2]; // Blue value

                // Convert RGB to grayscale using the luminosity method
                output_image[idx] = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
            }
        }
    }

    void convertFloatToUChar(const float* input_image_data, unsigned char* output_image, int size_channels) {
        #pragma omp parallel for
        for (int i = 0; i < size_channels; i++) {
            output_image[i] = static_cast<unsigned char>(255 * input_image_data[i]);
        }
    }

    void computeHistogram(const unsigned char* gray_image, int* histogram, int size) {
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            int value = gray_image[i];
            //TUDO: Existe uma alternativa para isto
            #pragma omp atomic
            histogram[value]++;
        }
    }
    void computeHistogramWithCUB(const unsigned char* d_input, int* d_histogram, size_t num_items, cudaStream_t stream = 0) {
        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;

        // Get amount of temporary storage needed
        cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                                            d_input, d_histogram,
                                            HISTOGRAM_LENGTH + 1, 0, 256, num_items, stream);

        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        // Compute the histogram
        cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                                            d_input, d_histogram,
                                            HISTOGRAM_LENGTH + 1, 0, 256, num_items, stream);

        // Free temporary storage
        cudaFree(d_temp_storage);
    }



    void computeCDF(int* histogram, float* cdf, int size, int total_pixels) {
        // Computação das probabilidades em paralelo
        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            cdf[i] = prob(histogram[i], total_pixels);
        }
        // Tentar paralelizar
        for (int i = 1; i < size; i++) {
            cdf[i] += cdf[i - 1];
        }
    }

    // Encontrar o valor minimo com o reduction
    float findMinCDF(const float* cdf, int length) {
        float cdf_min = cdf[0];
        #pragma omp parallel for reduction(min: cdf_min)
        for (int i = 1; i < length; i++) {
            cdf_min = std::min(cdf_min, cdf[i]);
        }
        return cdf_min;
    }

    void applyColorCorrection(unsigned char* image, const float* cdf, float cdf_min, int size_channels) {
        #pragma omp parallel for
        for (int i = 0; i < size_channels; i++) {
            image[i] = correct_color(cdf[image[i]], cdf_min);
        }
    }

    void convertToFloat(const unsigned char* input_image, float* output_image_data, int size_channels) {
        #pragma omp parallel for
        for (int i = 0; i < size_channels; i++) {
            output_image_data[i] = static_cast<float>(input_image[i]) / 255.0f;
        }
    }

    static void histogram_equalization(const int width, const int height,
                                       const float *input_image_data,
                                       float *output_image_data,
                                       const std::shared_ptr<unsigned char[]> &uchar_image,
                                       const std::shared_ptr<unsigned char[]> &gray_image,
                                       int (&histogram)[HISTOGRAM_LENGTH],
                                       float (&cdf)[HISTOGRAM_LENGTH]) {

        constexpr auto channels = 3;
        const auto size = width * height;
        const auto size_channels = size * channels;


        convertFloatToUChar(input_image_data, uchar_image.get(), size_channels);

        convertToGrayscale(uchar_image.get(), gray_image.get(), width, height);

        std::fill(histogram, histogram + HISTOGRAM_LENGTH, 0);



        // Allocate GPU memory
        unsigned char *d_gray_image;
        int *d_histogram;
        cudaMalloc(&d_gray_image, size * sizeof(unsigned char));
        cudaMalloc(&d_histogram, (HISTOGRAM_LENGTH + 1) * sizeof(int));
        cudaCheckError();

        // Copy grayscale image to GPU
        cudaMemcpy(d_gray_image, gray_image.get(), size * sizeof(unsigned char), cudaMemcpyHostToDevice);
        cudaCheckError();

        // Compute histogram on GPU
        computeHistogramWithCUB(d_gray_image, d_histogram, size);

        // Copy histogram back to host
        cudaMemcpy(histogram, d_histogram, (HISTOGRAM_LENGTH + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        cudaCheckError();

        // Free GPU memory
        cudaFree(d_gray_image);
        cudaFree(d_histogram);





        cdf[0] = prob(histogram[0], size);
        computeCDF(histogram, cdf, HISTOGRAM_LENGTH, size*channels);

        //auto cdf_min = cdf[0];
        float cdf_min = findMinCDF(cdf, HISTOGRAM_LENGTH);

        applyColorCorrection(uchar_image.get(), cdf, cdf_min, size_channels);

        convertToFloat(uchar_image.get(), output_image_data, size_channels);
    }



    wbImage_t iterative_histogram_equalization(wbImage_t &input_image, int iterations) {

        const auto width = wbImage_getWidth(input_image);
        const auto height = wbImage_getHeight(input_image);
        constexpr auto channels = 3;
        const auto size = width * height;
        const auto size_channels = size * channels;

        wbImage_t output_image = wbImage_new(width, height, channels);
        float *input_image_data = wbImage_getData(input_image);
        float *output_image_data = wbImage_getData(output_image);

        std::shared_ptr<unsigned char[]> uchar_image(new unsigned char[size_channels]);
        std::shared_ptr<unsigned char[]> gray_image(new unsigned char[size]);

        int histogram[HISTOGRAM_LENGTH];
        float cdf[HISTOGRAM_LENGTH];

        for (int i = 0; i < iterations; i++) {
            histogram_equalization(width, height,
                                   input_image_data, output_image_data,
                                   uchar_image, gray_image,
                                   histogram, cdf);

            input_image_data = output_image_data;
        }

        return output_image;
    }
}
