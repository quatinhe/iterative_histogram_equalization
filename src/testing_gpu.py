import subprocess
import os
import csv
import time
from PIL import Image


images = ['../dataset/output2000x2000.ppm', '../dataset/output3000x3000.ppm', '../dataset/output4000x4000.ppm', '../dataset/output7000x7000.ppm', '../dataset/output8500x8500.ppm', '../dataset/output10000x10000.ppm', '../dataset/big_sample.ppm', '../dataset/borabora_1.ppm', '../dataset/input01.ppm']
threads = [1, 4, 8, 16]
iterations = 500


output_dir = '../src/'
csv_file = 'performance_comparison.csv'


def run_command(command):
    start = time.time()
    subprocess.run(command, shell=True)
    end = time.time()
    return end - start


def get_image_size(image_path):
    with Image.open(image_path) as img:
        return f"{img.width}x{img.height}"


with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Size', 'Threads', 'GPU Time (s)', 'CPU Time (s)'])


    for img in images:
        image_size = get_image_size(img)
        for thread in threads:

            gpu_command = f'../cmake-build-debug/project_par_gpu {img} {iterations} {os.path.join(output_dir, "output_gpu.ppm")}'
            cpu_command = f'OMP_NUM_THREADS={thread} ../cmake-build-debug/project_par {img} {iterations} {os.path.join(output_dir, "output_cpu.ppm")}'


            gpu_time = run_command(gpu_command)
            cpu_time = run_command(cpu_command)


            writer.writerow([image_size, thread, gpu_time, cpu_time])
            print(f'Tested {image_size} with {thread} threads: GPU = {gpu_time}s, CPU = {cpu_time}s')
