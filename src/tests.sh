#!/bin/bash

set -e


results_file="results.csv"
echo "test_type,size,iterations,time_seconds" > "$results_file"

echo "Cleaning up old outputs..."
rm -rf outputs
mkdir -p outputs

echo "Compiling the project..."
g++ -fopenmp -I../include -I../libwb -o histogram_eq_seq main.cpp histogram_eq.cpp $(find ../libwb -name "*.cpp" ! -name "*_test.cpp") -lm
g++ -fopenmp -I../include -I../libwb -o histogram_eq_par main.cpp histogram_eq_par.cpp $(find ../libwb -name "*.cpp" ! -name "*_test.cpp") -lm

sizes=(500 1000 1500)
iterations=(1 2 3)
image_files=("../src/lake.ppm" "../src/lake.ppm" "../src/lake.ppm")


for size in "${sizes[@]}"; do
    for iter in "${iterations[@]}"; do
        for image_file in "${image_files[@]}"; do
            output_file_seq="outputs/output_seq_${size}_${iter}.ppm"
            output_file_par="outputs/output_par_${size}_${iter}.ppm"

            echo "Testing image: $image_file with size: $size and iterations: $iter"


            start_time=$(date +%s.%N)
            ./histogram_eq_seq "$image_file" "$iter" "$output_file_seq"
            end_time=$(date +%s.%N)
            time_seq=$(echo "$end_time - $start_time" | bc)
            echo "Sequential version completed for $size and $iter in $time_seq seconds."


            echo "sequential,$size,$iter,$time_seq" >> "$results_file"


            start_time=$(date +%s.%N)
            ./histogram_eq_par "$image_file" "$iter" "$output_file_par"
            end_time=$(date +%s.%N)
            time_par=$(echo "$end_time - $start_time" | bc)
            echo "Parallel version completed for $size and $iter in $time_par seconds."


            echo "parallel,$size,$iter,$time_par" >> "$results_file"

            sleep 1
        done
    done
done

echo "All tests completed."
