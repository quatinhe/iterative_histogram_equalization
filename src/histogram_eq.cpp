//
// Created by herve on 13-04-2024.
//

#include "histogram_eq.h"
#include <chrono>
#include <iostream>
#include <algorithm>

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

        //auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < size_channels; i++) //converte float para uchar (Leva aproximadamente 0.03s para o lake.ppm)
            uchar_image[i] = (unsigned char) (255 * input_image_data[i]);
        //auto finish = std::chrono::high_resolution_clock::now();
            //std::chrono::duration<double> elapsed = finish - start;
            //std::cout << "Converter float to Uchar time: " << elapsed.count() << " s\n";

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < height; i++) //converte para tons de cinza (Leva aproximadamente 0.05s para o lake.ppm)
            for (int j = 0; j < width; j++) {
                auto idx = i * width + j;
                auto r = uchar_image[3 * idx];
                auto g = uchar_image[3 * idx + 1];
                auto b = uchar_image[3 * idx + 2];
                gray_image[idx] = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
            }


        std::fill(histogram, histogram + HISTOGRAM_LENGTH, 0); //inicializa histograma

        for (int i = 0; i < size; i++) //calcula histograma
            histogram[gray_image[i]]++; //verificar se é necessario usar atomic

        cdf[0] = prob(histogram[0], size); //calcula cdf
        for (int i = 1; i < HISTOGRAM_LENGTH; i++)
            cdf[i] = cdf[i - 1] + prob(histogram[i], size); //nao paralelizavel

        auto cdf_min = cdf[0];
        #pragma omp parallel for reduction(min: cdf_min)
        for (int i = 1; i < HISTOGRAM_LENGTH; i++)
            cdf_min = std::min(cdf_min, cdf[i]);

        #pragma omp parallel for
        for (int i = 0; i < size_channels; i++) //correção de cor
            uchar_image[i] = correct_color(cdf[uchar_image[i]], cdf_min);

        for (int i = 0; i < size_channels; i++) //converte uchar para float
            output_image_data[i] = static_cast<float>(uchar_image[i]) / 255.0f;
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