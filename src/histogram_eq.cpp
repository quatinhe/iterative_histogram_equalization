//
// Created by herve on 13-04-2024.
//

#include "histogram_eq.h"
#include <iostream>
#include <chrono>
#include <algorithm>

namespace cp {
    constexpr auto HISTOGRAM_LENGTH = 256;

    static float inline prob(const int x, const int size) {
        return (float) x / (float) size;
    }

    static unsigned char inline clamp(unsigned char x) {
        return std::min(std::max(x, static_cast<unsigned char>(0)), static_cast<unsigned char>(255));
    }

    static unsigned char inline correct_color(float cdf_val, float cdf_min) {
        return clamp(static_cast<unsigned char>(255 * (cdf_val - cdf_min) / (1 - cdf_min)));
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

        // Section 1: Image Conversion
        auto start1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < size_channels; i++)
            uchar_image[i] = (unsigned char) (255 * input_image_data[i]);

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++) {
                auto idx = i * width + j;
                auto r = uchar_image[3 * idx];
                auto g = uchar_image[3 * idx + 1];
                auto b = uchar_image[3 * idx + 2];
                gray_image[idx] = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
            }
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed1 = end1 - start1;
        double milliseconds1 = elapsed1.count() * 1000.0;
        std::cout << "Elapsed time for image conversion: " << milliseconds1 << " milliseconds" << std::endl;

        // Section 2: Histogram Calculation
        auto start2 = std::chrono::high_resolution_clock::now();
        std::fill(histogram, histogram + HISTOGRAM_LENGTH, 0);
        for (int i = 0; i < size; i++)
            histogram[gray_image[i]]++;

        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed2 = end2 - start2;
        double milliseconds2 = elapsed2.count() * 1000.0;
        std::cout << "Elapsed time for histogram calculation: " << milliseconds2 << " milliseconds" << std::endl;

        // Section 3: CDF Calculation
        auto start3 = std::chrono::high_resolution_clock::now();
        cdf[0] = prob(histogram[0], size);
        for (int i = 1; i < HISTOGRAM_LENGTH; i++)
            cdf[i] = cdf[i - 1] + prob(histogram[i], size);

        auto cdf_min = cdf[0];
        for (int i = 1; i < HISTOGRAM_LENGTH; i++)
            cdf_min = std::min(cdf_min, cdf[i]);

        auto end3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed3 = end3 - start3;
        double milliseconds3 = elapsed3.count() * 1000.0;
        std::cout << "Elapsed time for CDF calculation: " << milliseconds3 << " milliseconds" << std::endl;

        // Section 4: Image Correction and Output
        auto start4 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < size_channels; i++)
            uchar_image[i] = correct_color(cdf[uchar_image[i]], cdf_min);

        for (int i = 0; i < size_channels; i++)
            output_image_data[i] = static_cast<float>(uchar_image[i]) / 255.0f;

        auto end4 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed4 = end4 - start4;
        double milliseconds4 = elapsed4.count() * 1000.0;
        std::cout << "Elapsed time for image correction and output: " << milliseconds4 << " milliseconds" << std::endl;
    }

    wbImage_t iterative_histogram_equalization(wbImage_t &input_image, int iterations) {

        auto startTotal = std::chrono::high_resolution_clock::now();

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
        auto endTotal = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> totalElapsed = endTotal - startTotal;
        double milliseconds = totalElapsed.count() * 1000.0;
        std::cout << "Elapsed time: " << milliseconds << " milliseconds" << std::endl;

        return output_image;
    }
}