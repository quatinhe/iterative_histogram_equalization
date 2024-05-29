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
    constexpr auto NUM_THREADS = 8;


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
<<<<<<< Updated upstream
#pragma omp parallel for collapse(2) num_threads(NUM_THREADS)
=======
#pragma omp parallel for collapse(2)
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
        #pragma omp parallel for num_threads(NUM_THREADS)
=======
#pragma omp parallel for
>>>>>>> Stashed changes
        for (int i = 0; i < size_channels; i++) {
            output_image[i] = static_cast<unsigned char>(255 * input_image_data[i]);
        }
    }

    void computeCDF(int* histogram, float* cdf, int size, int total_pixels) {
        // Computação das probabilidades em paralelo
<<<<<<< Updated upstream
        #pragma omp parallel for num_threads(NUM_THREADS)
=======
#pragma omp parallel for
>>>>>>> Stashed changes
        for (int i = 0; i < size; i++) {
            cdf[i] = prob(histogram[i], total_pixels);
        }
        for (int i = 1; i < size; i++) {
            cdf[i] += cdf[i - 1];
        }
    }

    void applyColorCorrectionAndConvertToFloat(const unsigned char* image, const float* cdf, float cdf_min, int size_channels, float* output_image_data) {
<<<<<<< Updated upstream
#pragma omp parallel for num_threads(NUM_THREADS)
=======
#pragma omp parallel for
>>>>>>> Stashed changes
        for (int i = 0; i < size_channels; i++) {
            float color = correct_color(cdf[image[i]], cdf_min);
            output_image_data[i] = color / 255.0f;
        }
    }

    void computeHistogram(const unsigned char* gray_image, int* histogram, int size) {
<<<<<<< Updated upstream
        #pragma omp parallel for reduction(+ : histogram[:HISTOGRAM_LENGTH]) num_threads(NUM_THREADS)
=======
#pragma omp parallel for reduction(+ : histogram[:HISTOGRAM_LENGTH])
>>>>>>> Stashed changes
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
                                       float (&cdf)[HISTOGRAM_LENGTH]) {

        constexpr auto channels = 3;
        const auto size = width * height;
        const auto size_channels = size * channels;

        //Section 1 - Convert the input image to grayscale
        auto start = std::chrono::high_resolution_clock::now();

        convertFloatToUChar(input_image_data, uchar_image.get(), size_channels);

        convertToGrayscale(uchar_image.get(), gray_image.get(), width, height);

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        //std::cout << "Tempo total image conversion : " << elapsed.count() << "\n";


        //Section 2 - Compute the histogram
        auto start2 = std::chrono::high_resolution_clock::now();

        std::fill(histogram, histogram + HISTOGRAM_LENGTH, 0);
        computeHistogram(gray_image.get(), histogram, size);

        auto finish2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed2 = finish2 - start2;
        //std::cout << "Tempo total histogram : " << elapsed2.count() << "\n";

        //Section 3 - Compute the CDF
        auto start3 = std::chrono::high_resolution_clock::now();
        cdf[0] = prob(histogram[0], size);
        computeCDF(histogram, cdf, HISTOGRAM_LENGTH, size*channels);
        auto finish3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed3 = finish3 - start3;
        //std::cout << "Tempo total cdf : " << elapsed3.count() << "\n";
        //auto cdf_min = cdf[0];


        //Section 4 - Apply the color correction
        auto start4 = std::chrono::high_resolution_clock::now();
        float cdf_min = cdf[0];
        applyColorCorrectionAndConvertToFloat(uchar_image.get(), cdf, cdf_min, size_channels, output_image_data);
        auto finish4 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed4 = finish4 - start4;
        //std::cout << "Tempo total color correction : " << elapsed4.count() << "\n";
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