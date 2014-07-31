#!/bin/bash
icc -mmic -o sobPHI -openmp sobelParallelCPU.c -lm
#mv sobOpt OptsobMKLv4
#scp sobMKLv4 mic0:
scp phi_script.sh mic0:
scp sobPHI mic0:
#ssh mic0
