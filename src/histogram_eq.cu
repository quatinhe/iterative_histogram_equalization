//
// Created by quatinhe on 5/7/24.
//

#include "histogram_eq.h"
#include <chrono>
#include <iostream>
#include <omp.h>
#include <cub/cub.cuh>
#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <algorithm>


#define HISTOGRAM_LENGTH 256

#define cudaCheckError() {                                           \
    cudaError_t e = cudaGetLastError();                              \
    if (e != cudaSuccess) {                                          \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
}

__device__ unsigned char clamp_device(unsigned char x) {
    return min(max(x, static_cast<unsigned char>(0)), static_cast<unsigned char>(255));
}

__device__ unsigned char correct_color_device(float cdf_val, float cdf_min) {
    return clamp_device(static_cast<unsigned char>(255 * (cdf_val - cdf_min) / (1 - cdf_min)));
}

// Transforma os valores do histograme prob (preparar para o cdf)
__global__ void transformHistogramToProb(const int* histogram, float* prob, int total_pixels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < HISTOGRAM_LENGTH) {
        prob[i] = static_cast<float>(histogram[i]) / total_pixels;
    }
}

__global__ void convertToGrayscaleKernel(const unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        unsigned char r = input[3 * idx];
        unsigned char g = input[3 * idx + 1];
        unsigned char b = input[3 * idx + 2];

        // (Método da limunisidade) --> para a conversão grayscale
        unsigned char gray = static_cast<unsigned char>(0.21f * r + 0.71f * g + 0.07f * b);
        output[idx] = gray;
    }
}

__global__ void applyColorCorrectionKernel(unsigned char* image, const float* cdf, float cdf_min, int size_channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size_channels) {
        unsigned char pixelValue = image[idx];
        image[idx] = correct_color_device(cdf[pixelValue], cdf_min);
    }
}

__global__ void convertFloatToUCharKernel(const float* input, unsigned char* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = static_cast<unsigned char>(255 * input[idx]);
    }
}

__global__ void convertToFloatKernel(const unsigned char* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] / 255.0f;
    }
}


namespace cp {

    static float inline prob(const int x, const int size) {
        return (float) x / (float) size;
    }

    static unsigned char inline clamp(unsigned char x) { //assegura q um valor está no range
        return std::min(std::max(x, static_cast<unsigned char>(0)), static_cast<unsigned char>(255));
    }

    static unsigned char inline correct_color(float cdf_val, float cdf_min) {
        return clamp(static_cast<unsigned char>(255 * (cdf_val - cdf_min) / (1 - cdf_min)));
    }


    void convertToGrayscaleGPU(const unsigned char* d_input, unsigned char* d_output, int width, int height) {
        //cada bloco contÊm 16x16=256 threads  --> Fazer testes para ver se o 16 é mesmo o melhor
        dim3 block(16, 16);
        //Especifica o número de blocos em cada dimensão (widht e height)
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        convertToGrayscaleKernel<<<grid, block>>>(d_input, d_output, width, height);
        cudaCheckError();
    }

    void convertFloatToUChar(const float* input_image_data, unsigned char* output_image, int size_channels) {
        #pragma omp parallel for
        for (int i = 0; i < size_channels; i++) {
            output_image[i] = static_cast<unsigned char>(255 * input_image_data[i]);
        }
    }

    void convertFloatToUCharGPU(const float* d_input, unsigned char* d_output, int size_channels) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (size_channels + threadsPerBlock - 1) / threadsPerBlock;

        convertFloatToUCharKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size_channels);
        cudaCheckError();
    }



    void computeHistogramWithCUB(const unsigned char* d_input, int* d_histogram, size_t num_items, cudaStream_t stream = 0) {
        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;

        // GET do tempo de storage necessário
        cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                                            d_input, d_histogram,
                                            HISTOGRAM_LENGTH + 1, 0, 256, num_items, stream);

        // Alocar a storage temp.
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        // Computação do histograma
        cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                                            d_input, d_histogram,
                                            HISTOGRAM_LENGTH + 1, 0, 256, num_items, stream);

        // Libertar storage temporária
        cudaFree(d_temp_storage);
    }


    void computeCDFWithCUB(const int* d_histogram, float* d_cdf, int num_bins, int total_pixels, cudaStream_t stream = 0) {
        float* d_prob;
        cudaMalloc(&d_prob, num_bins * sizeof(float));
        cudaCheckError();

        // Converter histograma paara prob no GPU
        int threadsPerBlock = 256;
        int numBlocks = (num_bins + threadsPerBlock - 1) / threadsPerBlock;
        transformHistogramToProb<<<numBlocks, threadsPerBlock>>>(d_histogram, d_prob, total_pixels);
        cudaCheckError();

        // storage temporaria para o profixo sum
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_prob, d_cdf, num_bins, stream);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_prob, d_cdf, num_bins, stream);
        cudaCheckError();

        cudaFree(d_temp_storage);
        cudaFree(d_prob);
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

    void applyColorCorrectionGPU(unsigned char* d_image, const float* d_cdf, float cdf_min, int size_channels) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (size_channels + threadsPerBlock - 1) / threadsPerBlock;

        applyColorCorrectionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_cdf, cdf_min, size_channels);
        cudaCheckError();
    }

    void convertToFloat(const unsigned char* input_image, float* output_image_data, int size_channels) {
        #pragma omp parallel for
        for (int i = 0; i < size_channels; i++) {
            output_image_data[i] = static_cast<float>(input_image[i]) / 255.0f;
        }
    }
    void convertToFloatGPU(const unsigned char* d_input, float* d_output, int size_channels) {
        int threadsPerBlock = 256;
        int blocksPerGrid = (size_channels + threadsPerBlock - 1) / threadsPerBlock;

        convertToFloatKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size_channels);
        cudaCheckError();
    }


    static void histogram_equalization(
            const int width, const int height,
            float *d_input_image_data,
            float *d_output_image_data,
            unsigned char *d_input_image,
            unsigned char *d_gray_image,
            unsigned char *d_uchar_image,
            int *d_histogram,
            float *d_cdf,
            float (&cdf)[HISTOGRAM_LENGTH]
    ) {

        constexpr auto channels = 3;
        const auto size = width * height;
        const auto size_channels = size * channels;

        convertFloatToUCharGPU(d_input_image_data, d_uchar_image, size_channels);

        convertToGrayscaleGPU(d_input_image, d_gray_image, width, height);

        cudaMemset(d_histogram, 0, HISTOGRAM_LENGTH * sizeof(int));

        computeHistogramWithCUB(d_gray_image, d_histogram, size);

        computeCDFWithCUB(d_histogram, d_cdf, HISTOGRAM_LENGTH, size);

        cudaMemcpy(cdf, d_cdf, HISTOGRAM_LENGTH * sizeof(float), cudaMemcpyDeviceToHost);
        cudaCheckError();

        float cdf_min = *std::min_element(cdf, cdf + HISTOGRAM_LENGTH);


        applyColorCorrectionGPU(d_uchar_image, d_cdf, cdf_min, size_channels);

        convertToFloatGPU(d_uchar_image, d_output_image_data, size_channels);
    }




    wbImage_t iterative_histogram_equalization(wbImage_t &input_image, int iterations) {
        const int width = wbImage_getWidth(input_image);
        const int height = wbImage_getHeight(input_image);
        constexpr int channels = 3;
        const int size = width * height;
        const int size_channels = size * channels;

        // Allocate memory for the output image
        wbImage_t output_image = wbImage_new(width, height, channels);
        float *output_image_data = wbImage_getData(output_image);

        // GPU memory allocations
        float *d_input_image_data, *d_output_image_data;
        cudaMalloc(&d_input_image_data, size_channels * sizeof(float));
        cudaMalloc(&d_output_image_data, size_channels * sizeof(float));

        // Transfer input data to GPU
        cudaMemcpy(d_input_image_data, wbImage_getData(input_image), size_channels * sizeof(float), cudaMemcpyHostToDevice);

        unsigned char *d_input_image, *d_gray_image, *d_uchar_image;
        int *d_histogram;
        float *d_cdf;
        cudaMalloc(&d_input_image, size_channels * sizeof(unsigned char));
        cudaMalloc(&d_gray_image, size * sizeof(unsigned char));
        cudaMalloc(&d_uchar_image, size_channels * sizeof(unsigned char));
        cudaMalloc(&d_histogram, HISTOGRAM_LENGTH * sizeof(int));
        cudaMalloc(&d_cdf, HISTOGRAM_LENGTH * sizeof(float));
        float cdf[256];
        // Processing iterations
        for (int i = 0; i < iterations; i++) {
            histogram_equalization(width, height,
                                   d_input_image_data, d_output_image_data,
                                   d_input_image, d_gray_image, d_uchar_image,
                                   d_histogram, d_cdf,
                                   cdf);

            // Swap pointers for next iteration
            std::swap(d_input_image_data, d_output_image_data);
        }

        // Copy final output data back to host
        cudaMemcpy(output_image_data, d_output_image_data, size_channels * sizeof(float), cudaMemcpyDeviceToHost);

        // Free all GPU memory
        cudaFree(d_input_image);
        cudaFree(d_gray_image);
        cudaFree(d_uchar_image);
        cudaFree(d_histogram);
        cudaFree(d_cdf);
        cudaFree(d_input_image_data);
        cudaFree(d_output_image_data);

        return output_image;
    }
}
