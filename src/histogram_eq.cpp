//
// Created by herve on 13-04-2024.
//

#include "histogram_eq.h"
#include <algorithm>
#include <chrono>


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
                                       float (&cdf)[HISTOGRAM_LENGTH],
                                       double (&chronos)[5]) {

        constexpr auto channels = 3;
        const auto size = width * height;
        const auto size_channels = size * channels;

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < size_channels; i++)
            uchar_image[i] = (unsigned char) (255 * input_image_data[i]);
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed1 = finish - start;
        chronos[0] += elapsed1.count();

        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++) {
                auto idx = i * width + j;
                auto r = uchar_image[3 * idx];
                auto g = uchar_image[3 * idx + 1];
                auto b = uchar_image[3 * idx + 2];
                gray_image[idx] = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
            }

        finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed2 = finish - start;
        chronos[1] += elapsed2.count();

        std::fill(histogram, histogram + HISTOGRAM_LENGTH, 0);
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < size; i++)
            histogram[gray_image[i]]++;
        finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed3 = finish - start;
        chronos[2] += elapsed3.count();

        cdf[0] = prob(histogram[0], size);

        start = std::chrono::high_resolution_clock::now();
        for (int i = 1; i < HISTOGRAM_LENGTH; i++)
            cdf[i] = cdf[i - 1] + prob(histogram[i], size);

        finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed4 = finish - start;
        chronos[3] += elapsed4.count();

        auto cdf_min = cdf[0];
        for (int i = 1; i < HISTOGRAM_LENGTH; i++)
            cdf_min = std::min(cdf_min, cdf[i]);

        start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < size_channels; i++)
            uchar_image[i] = correct_color(cdf[uchar_image[i]], cdf_min);

        for (int i = 0; i < size_channels; i++)
            output_image_data[i] = static_cast<float>(uchar_image[i]) / 255.0f;
        finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed6 = finish - start;
        chronos[4] += elapsed6.count();


    }

    wbImage_t iterative_histogram_equalization(wbImage_t &input_image, int iterations) {
        double chronos[5] = {};
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
                                   histogram, cdf,
                                   chronos);

            input_image_data = output_image_data;
        }

        for(int i = 0; i < 5; i++){
            std::cout << "Tempo total função " << i << " : " << chronos[i] << "\n";
        }
        return output_image;
    }
}