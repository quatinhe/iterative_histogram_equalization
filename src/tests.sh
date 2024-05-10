#!/bin/bash
# Script to compile and test the Histogram Equalization project

# Set script to exit on any errors.
set -e

# Clean up previous outputs
rm -f outputs/output.txt
mkdir -p outputs
touch outputs/output.txt

# Compile the project
echo "Compiling the project..."
g++ -fopenmp -I../include -I../libwb -o histogram_eq main.cpp histogram_eq.cpp $(find ../libwb -name "*.cpp" ! -name "*_test.cpp") -lm  # Assuming main.cpp contains your entire project code

# Define test parameters
image_sizes=(100 100 100 200 200 300)
iterations=(1 2 3 4 5)

# Assuming you have test images stored in a directory "test_images"
image_files=("../src/lake.ppm" "../src/lake.ppm" "../src/lake.ppm")

# Loop through the test images and iterations
for image_file in "${image_files[@]}"; do
    for size in "${image_sizes[@]}"; do
        for iter in "${iterations[@]}"; do
            output_file="outputs/output_${size}_${iter}.ppm"
            echo "Testing image: $image_file with size: $size and iterations: $iter"
            echo "Output will be saved to $output_file"

            # Run the histogram equalization
            ./histogram_eq "$image_file" "$iter" "$output_file" >> outputs/output.txt

            # Log completion
            echo "Test with size $size and iterations $iter completed." >> outputs/output.txt
            sleep 1  # Optional: Sleep to pace the tests
        done
    done
done

echo "All tests completed."
