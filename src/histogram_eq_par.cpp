//
// Created by quatinhe on 5/7/24.
//

#include "histogram_eq.h"
#include <chrono>
#include <iostream>
#include <algorithm>
#include <omp.h>

namespace cp {
    constexpr auto HISTOGRAM_LENGTH = 256; // Corresponde ao número de tons de cinza em 8bit
    constexpr auto NUM_THREADS = 16;


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
#pragma omp for collapse(2)
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
#pragma omp for
        for (int i = 0; i < size_channels; i++) {
            output_image[i] = static_cast<unsigned char>(255 * input_image_data[i]);
        }
    }

    void computeCDF(int* histogram, float* cdf, int size, int total_pixels) {
#pragma omp single
        for (int i = 0; i < size; i++) {
            cdf[i] = cdf[i - 1] + prob(histogram[i], total_pixels);
        }

    }

    void applyColorCorrectionAndConvertToFloat(const unsigned char* image, const float* cdf, float cdf_min, int size_channels, float* output_image_data) {
#pragma omp for
        for (int i = 0; i < size_channels; i++) {
            float color = correct_color(cdf[image[i]], cdf_min);
            output_image_data[i] = color / 255.0f;
        }
    }

    void computeHistogram(const unsigned char* gray_image, int* histogram, int size) {
#pragma omp for reduction(+ : histogram[:HISTOGRAM_LENGTH])
        for (int i = 0; i < size; i++) {
            histogram[gray_image[i]]++;
        }
    }

    static void histogram_equalization(const int width, const int height,
                                       const float *input_image_data,
                                       float *output_image_data,
                                       const std::shared_ptr<unsigned char[]> &uchar_image,
                                       const std::shared_ptr<unsigned char[]> &gray_image,
                                       int (&histogram)[HISTOGRAM_LENGTH],
                                       float (&cdf)[HISTOGRAM_LENGTH],
                                       double (&chronos)[5]) {

        constexpr auto channels = 3;
        const auto size = width * height;
        const auto size_channels = size * channels;

        //Section 1 - Convert the input image to grayscale
        auto start = std::chrono::high_resolution_clock::now();
        convertFloatToUChar(input_image_data, uchar_image.get(), size_channels);
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed1 = finish - start;
        chronos[0] += elapsed1.count();

        start = std::chrono::high_resolution_clock::now();
        convertToGrayscale(uchar_image.get(), gray_image.get(), width, height);
        finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed2 = finish - start;
        chronos[1] += elapsed2.count();


        //Section 2 - Compute the histogram

        std::fill(histogram, histogram + HISTOGRAM_LENGTH, 0);
        start = std::chrono::high_resolution_clock::now();
        computeHistogram(gray_image.get(), histogram, size);
        finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed3 = finish - start;
        chronos[2] += elapsed3.count();

        //Section 3 - Compute the CDF
        cdf[0] = prob(histogram[0], size);
        start = std::chrono::high_resolution_clock::now();
        computeCDF(histogram, cdf, HISTOGRAM_LENGTH, size*channels);
        finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed4 = finish - start;
        chronos[3] += elapsed4.count();

        //Section 4 - Apply the color correction
        float cdf_min = cdf[0];
        start = std::chrono::high_resolution_clock::now();
        applyColorCorrectionAndConvertToFloat(uchar_image.get(), cdf, cdf_min, size_channels, output_image_data);
        finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed5 = finish - start;

        chronos[4] += elapsed5.count();

    }


    wbImage_t iterative_histogram_equalization(wbImage_t &input_image, int iterations) {
        double chronos[5] = {};
        double times[5] = {6.57, 4.30, 1.83, 0.00018, 26.13};
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
#pragma omp parallel num_threads (NUM_THREADS)
            {
                histogram_equalization(width, height,
                                       input_image_data, output_image_data,
                                       uchar_image, gray_image,
                                       histogram, cdf, chronos);

                input_image_data = output_image_data;
            }
        }
        for(int i = 0; i < 5; i++){
            std::cout << "Tempo total função " << i << " : " << chronos[i] / (NUM_THREADS) << "\n";
            std::cout << "Speedup " << i << " : " << times[i]/(chronos[i] / (NUM_THREADS)) << "\n";
            std::cout << "Efficiency " << i << " : " << (times[i]/(chronos[i] / (NUM_THREADS)) / NUM_THREADS) << "\n";

        }
        return output_image;
    }
}
