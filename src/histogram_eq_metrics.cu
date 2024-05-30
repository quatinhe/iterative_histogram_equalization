//
// Created by quatinhe on 5/7/24.
//

#include "histogram_eq.h"
#include <iostream>
#include <cub/cub.cuh>
#include <thrust/transform.h>
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

__global__ void applyColorCorrectionKernel(unsigned char* image, const float* cdf, float cdf_min, int size_channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size_channels) {
        unsigned char pixelValue = image[idx];
        image[idx] = correct_color_device(cdf[pixelValue], cdf_min);
    }
}

__global__ void convertToFloatKernel(const unsigned char* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] / 255.0f;
    }
}

__global__ void combinedKernel(const float* inputFloat, unsigned char* outputGray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {

        unsigned char ucharValue = static_cast<unsigned char>(255 * inputFloat[idx]);



        unsigned char r = ucharValue;
        unsigned char g = ucharValue;
        unsigned char b = ucharValue;


        unsigned char gray = static_cast<unsigned char>(0.21f * r + 0.71f * g + 0.07f * b);
        outputGray[idx] = gray;
    }
}


namespace cp {

    void combinedGPU(const float* d_input, unsigned char* d_output, int width, int height, int blockWidth, int blockHeight) {
        dim3 block(blockWidth, blockHeight);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        combinedKernel<<<grid, block>>>(d_input, d_output, width, height);
        cudaCheckError();
    }



    void computeHistogramWithCUB(const unsigned char* d_input, int* d_histogram, size_t num_items, cudaStream_t stream = 0) {
        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;


        cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                                            d_input, d_histogram,
                                            HISTOGRAM_LENGTH + 1, 0, 256, num_items, stream);


        cudaMalloc(&d_temp_storage, temp_storage_bytes);


        cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                                            d_input, d_histogram,
                                            HISTOGRAM_LENGTH + 1, 0, 256, num_items, stream);


        cudaFree(d_temp_storage);
    }


    void computeCDFWithCUB(const int* d_histogram, float* d_cdf, float* d_prob, void* d_temp_storage, size_t temp_storage_bytes, int num_bins, int total_pixels, int blockWidth, int blockHeight, cudaStream_t stream = 0) {
        cudaCheckError();

        int threadsPerBlock = blockWidth * blockHeight;
        int numBlocks = (num_bins + threadsPerBlock - 1) / threadsPerBlock;
        transformHistogramToProb<<<numBlocks, threadsPerBlock>>>(d_histogram, d_prob, total_pixels);
        cudaCheckError();

        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_prob, d_cdf, num_bins, stream);
        cudaCheckError();
    }

    void applyColorCorrectionGPU(unsigned char* d_image, const float* d_cdf, float cdf_min, int size_channels, int blockWidth, int blockHeight) {
        int threadsPerBlock = blockWidth * blockHeight;
        int blocksPerGrid = (size_channels + threadsPerBlock - 1) / threadsPerBlock;

        applyColorCorrectionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_cdf, cdf_min, size_channels);
        cudaCheckError();
    }

    void convertToFloatGPU(const unsigned char* d_input, float* d_output, int size_channels, int blockWidth,int blockHeight) {
        int threadsPerBlock = blockWidth * blockHeight;
        int blocksPerGrid = (size_channels + threadsPerBlock - 1) / threadsPerBlock;

        convertToFloatKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, size_channels);
        cudaCheckError();
    }


    static void histogram_equalization(
            const int width, const int height,
            float *d_input_image_data,
            float *d_output_image_data,
            unsigned char *d_gray_image,
            unsigned char *d_uchar_image,
            int *d_histogram,
            float *d_cdf,
            float *d_prob,
            void *d_temp_storage,
            size_t temp_storage_bytes,
            cudaStream_t stream,
            float (&cdf)[HISTOGRAM_LENGTH]
    ) {

        constexpr auto channels = 3;
        const auto size = width * height;
        const auto size_channels = size * channels;
        int blockWidth = 16;
        int blockHeight = 16;

        combinedGPU(d_input_image_data, d_gray_image, width, height,  blockWidth, blockHeight);

        cudaMemset(d_histogram, 0, HISTOGRAM_LENGTH * sizeof(int));

        computeHistogramWithCUB(d_gray_image, d_histogram, size);

        computeCDFWithCUB(d_histogram, d_cdf, d_prob, d_temp_storage, temp_storage_bytes, HISTOGRAM_LENGTH, size, blockWidth, blockHeight, stream);


        float cdf_min = *std::min_element(cdf, cdf + HISTOGRAM_LENGTH);


        applyColorCorrectionGPU(d_uchar_image, d_cdf, cdf_min, size_channels, blockWidth, blockHeight);

        convertToFloatGPU(d_uchar_image, d_output_image_data, size_channels, blockWidth, blockHeight);
    }




    wbImage_t iterative_histogram_equalization(wbImage_t &input_image, int iterations) {
        const int width = wbImage_getWidth(input_image);
        const int height = wbImage_getHeight(input_image);
        constexpr int channels = 3;
        const int size = width * height;
        const int size_channels = size * channels;

        // alocar memoria para a imagem de output
        wbImage_t output_image = wbImage_new(width, height, channels);
        float *output_image_data = wbImage_getData(output_image);

        // alocacao de memoria GPU
        float *d_input_image_data, *d_output_image_data;
        cudaMalloc(&d_input_image_data, size_channels * sizeof(float));
        cudaMalloc(&d_output_image_data, size_channels * sizeof(float));

        // transferir o input para a GPU
        cudaMemcpy(d_input_image_data, wbImage_getData(input_image), size_channels * sizeof(float), cudaMemcpyHostToDevice);

        unsigned char *d_input_image, *d_gray_image, *d_uchar_image;
        int *d_histogram;
        float *d_cdf, *d_prob;
        void *d_temp_storage = nullptr; // storage temporaria para cub (cdf)
        size_t temp_storage_bytes = 0;

        cudaMalloc(&d_input_image, size_channels * sizeof(unsigned char));
        cudaMalloc(&d_gray_image, size * sizeof(unsigned char));
        cudaMalloc(&d_uchar_image, size_channels * sizeof(unsigned char));
        cudaMalloc(&d_histogram, HISTOGRAM_LENGTH * sizeof(int));
        cudaMalloc(&d_cdf, HISTOGRAM_LENGTH * sizeof(float));
        cudaMalloc(&d_prob, HISTOGRAM_LENGTH * sizeof(float));

        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_prob, d_cdf, HISTOGRAM_LENGTH);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        float cdf[256];

        for (int i = 0; i < iterations; i++) {
            histogram_equalization(width, height, d_input_image_data, d_output_image_data, d_gray_image, d_uchar_image,
                                   d_histogram, d_cdf, d_prob, d_temp_storage, temp_storage_bytes, nullptr, cdf);

            // trocar pointer para a prox iter
            std::swap(d_input_image_data, d_output_image_data);
        }


        cudaMemcpy(output_image_data, d_output_image_data, size_channels * sizeof(float), cudaMemcpyDeviceToHost);


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
