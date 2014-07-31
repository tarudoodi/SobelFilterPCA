/*
 ============================================================================
 Name        : sobel1D.cu
 Author      : Taru Doodi
 Version     : v1
 Contact	 : tarudoodi@ufl.edu / tarudoodi@gmail.com
 Copyright   : Your copyright notice
 Description : CUDA code for GPU K20x, specifically coded for [3x3] gradient masks.
 				runs for image sizes 2x2 to 2048x2048 and stores the timing data in .csv file.
 ============================================================================
 */

#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void sobel(int *d_input, int *d_output, long int N)
{
	//__shared__ int smBytes[];
	int idx = threadIdx.x;
	int bidx = blockIdx.x;
	float  sobel_x,sobel_y;
	int pixel = bidx*blockDim.x  + idx;
	if(N<=pixel && pixel<N*(N-1))
	{
		// the gradient in x direction
		sobel_x = (float) (1*(d_input[pixel -1 -N]) + 2*(d_input[pixel-N]) +1*(d_input[pixel-N+1]) - 1*(d_input[pixel +N-1]) - 2*(d_input[pixel+N]) - 1*(d_input[pixel+N+1]));
		// the gradient in y direction
		sobel_y = (float) (1*(d_input[pixel -1 -N]) - 1*(d_input[pixel -N+1]) + 2*(d_input[pixel-1]) - 2*(d_input[pixel +1]) + 1*(d_input[pixel -1 +N]) - 1*(d_input[pixel +1 +N]));
		/*
	// the gradient in x direction
	sobel_x = (double)( 1*(d_input[(idx-1)*N + idy-1]) +2*(d_input[idy-1 +idx*N]) +1*(d_input[idy-1 +N*(idx+1)]) - 1*(d_input[idy+1+N*(idx-1)]) - 2*(d_input[idy+1 + N*idx]) - 1*(d_input[idy+1 + N*(idx+1)]));
	// the gradient in y direction
	sobel_y = (double) (1*(d_input[idy-1 + N*(idx-1)]) - 1*(d_input[idy-1 +N*(idx+1)]) + 2*(d_input[idy + N*(idx-1)]) - 2*(d_input[idy +N*(idx+1)]) + 1*(d_input[idy+1 +N*(idx-1)]) - 1*(d_input[idy+1 + N*(idx+1)]));
		 */

	d_output[pixel] = (int) sqrt((sobel_x*sobel_x) + (sobel_y*sobel_y));
//d_output[pixel] =(int)  hypot(sobel_x, sobel_y);//	doesnt make any difference
if (d_output[pixel] > 255)
		{
			(d_output[pixel]) = 255;
		}
	}
    else
    {
        	d_output[pixel] = d_input[pixel];
    }
}


/* Initializing the image matrix */
void initializeImage(int *d_input,int N)
{
	for(int i=0; i<N*N; i++)
	{
		d_input[i] = (int) (rand()%255);
		//printf("%d the value of input array is %d \n", (i),h_input[i]);
	}
}

double timerval ()
{
	struct timeval st;
	gettimeofday (&st, NULL);
	return st. tv_sec + st. tv_usec * 1e-6;
}

int main()
{
	int i; // loop counters
	int m=0;
	int N=8; // rows, columns of image
	int *h_input,*h_output; //*h_x,*h_y;
	//double start_time, end_time;

	//CUDA Variables

	
	int *d_output,*d_input;
	int blockNum, threadNum;
	cudaEvent_t start,end;
	cudaEventCreate(&start); //timers
	cudaEventCreate(&end);
	float time;

	int kStart =2;
	int kStop =12;
	double seconds[kStop];
	int k =0;
//	k = kStop;
		for(k=kStart;k<=kStop; k++)
		{
			N = pow(2,k);
			h_input = (int*)malloc(N*N*sizeof(int));
			h_output = (int*)malloc(N*N*sizeof(int));
			// allocate GPU memory
			cudaMalloc((void**) &d_input, (N)*(N)*sizeof(int));
			cudaMalloc((void**) &d_output,(N)*(N)*sizeof(int));
			// generate the input array on the host
			//calculate thread number and block number
			if(N<=4)//8)//32)
			{
				blockNum = 1;
				threadNum =N*N;//128;//256;//512;// N*N;
			}
			else
			{
				blockNum = N*N/32;//64;//128;//256;//512;
				threadNum = 32; //64;//128;//256;//512;
			}
			initializeImage(h_input,N);
			cudaMemcpy(d_input, h_input, N*N*sizeof(int), cudaMemcpyHostToDevice); // transfer the array to the GPU
			cudaThreadSynchronize();
			cudaEventSynchronize(start);
			//start_time = timerval();
			cudaEventRecord(start,0);
			// launch the kernel
			for(i=0;i<1000;i++)
			{
//				printf("ran for ith = %d for size k =%d",i,k);
	//			sobel_edges(d_output, d_input, N); //just do the edges of the image on CPU as they
				//sobel<<<blockNum,(threadNum+2*N)*sizeof(int) >>>(d_input, d_output,N);
				sobel<<<blockNum,threadNum>>>(d_input, d_output,N);
				cudaThreadSynchronize();
			}
			cudaEventSynchronize(end);
			//end_time = timerval();
			cudaEventRecord(end,0);
			cudaMemcpy(h_output, d_output, N*N*sizeof(int), cudaMemcpyDeviceToHost);// copy back the result array to the CPU
			cudaEventElapsedTime(&time,start,end);
			//seconds[m++] = (start_time - end_time)/1000;
			seconds[m++] = time/1000;
			cudaFree(d_input);
			cudaFree(d_output);
			free(h_input);
			free(h_output);
		}

		// printing to file
		FILE *sobelParallelOutputfile;
		sobelParallelOutputfile = fopen("sobelParallelOutputfile.csv","a+");
		if(sobelParallelOutputfile == NULL)
		{
			printf("Could not open file\n");
			return EXIT_FAILURE;
		}
		fprintf(sobelParallelOutputfile,"N, Time taken \n");
		for(i=0;i<m;i++)
		{
			fprintf(sobelParallelOutputfile,"%lf,%f\n", pow(2,(i+kStart)), seconds[i]);
		}
		fclose(sobelParallelOutputfile); //Closing the file
		m=0; //reset m;
	return EXIT_SUCCESS;
}
