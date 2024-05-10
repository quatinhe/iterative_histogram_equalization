#!/bin/bash

set -e


results_file="results.csv"
echo "test_type,iterations,threads,time_seconds" > "$results_file"

echo "Limpar"
rm -rf outputs
mkdir -p outputs

echo "A compilar"
g++ -fopenmp -I../include -I../libwb -o histogram_eq_seq main.cpp histogram_eq.cpp $(find ../libwb -name "*.cpp" ! -name "*_test.cpp") -lm
g++ -fopenmp -I../include -I../libwb -o histogram_eq_par main.cpp histogram_eq_par.cpp $(find ../libwb -name "*.cpp" ! -name "*_test.cpp") -lm


iterations=(10 20 30 40 50 60 70)
threads=(4 6 8)
image_files=("../src/lake.ppm" "../src/lake.ppm" "../src/lake.ppm")


for iter in "${iterations[@]}"; do
    for num_threads in "${threads[@]}"; do
        for image_file in "${image_files[@]}"; do
            output_file_seq="outputs/output_seq_${iter}_${num_threads}.ppm"
            output_file_par="outputs/output_par_${iter}_${num_threads}.ppm"

            echo "Imagem a ser testada: $image_file Numero de iterações: $iter e threads: $num_threads"


            start_time=$(date +%s.%N)
            ./histogram_eq_seq "$image_file" "$iter" "$output_file_seq"
            end_time=$(date +%s.%N)
            time_seq=$(echo "$end_time - $start_time" | bc)
            echo "Versão sequencial completa para $iter iterações em $time_seq segundos."
            echo "sequential,$iter,1,$time_seq" >> "$results_file"

            export OMP_NUM_THREADS=$num_threads
            start_time=$(date +%s.%N)
            ./histogram_eq_par "$image_file" "$iter" "$output_file_par"
            end_time=$(date +%s.%N)
            time_par=$(echo "$end_time - $start_time" | bc)
            echo "Versão completa para $iter iterações e $num_threads threads em $time_par segundos."
            echo "parallel,$iter,$num_threads,$time_par" >> "$results_file"

            sleep 1
        done
    done
done

echo "Testes concluidos."
