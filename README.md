-We used the script "tests.sh" to collect data 
-Using the data obtained with "tests.sh" we used "testing.py" to plot a performance comparison by iteration count and thread number

-Since this first comparison wasn't enough to obtain concrete result we made the script "testing_gpu.py" to collect data not only on gpu and cpu performance but on several images aswell (this images were not put in github becasue they were too big but the script can be changed to run for example just one image).
-Using the data obtained with "testing_gpu.py" we used "plotting_gpu.py" to plot a graph by number of threads for cpu and images (We used a fixed optimal number of threads for GPU).

-All the .csv files obtained by the scripts are on anex in the final report.

-To calculate the initial measures we used an image 3000x3000 and 100 iterations
