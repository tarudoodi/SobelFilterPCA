#!/bin/bash
chmod +x *
export LD_LIBRARY_PATH=/root/home/Team4:$LD_LIBRARY_PATH
#for threads 2,4,8,16,32
./sobPHI
# for threads greater than 32
mv sobelParallelBenchmarking.csv sobel_xeon.csv
scp sobelParallelBenchmarking.csv 172.31.1.254:
logout
