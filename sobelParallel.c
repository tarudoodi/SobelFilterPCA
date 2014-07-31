/*
 ============================================================================
 Name        : sobelParallel.c
 Author      : Taru Doodi
 Version     : Parallel.2
 Contact	 : tarudoodi@ufl.edu / tarudoodi@gmail.com
 Copyright   : Your copyright notice
 Description : Parallel code for Sobel Filter using OpenMP, specifically coded for [3x3] gradient masks.
 ============================================================================
 */

#include <math.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

double sobel_x, sobel_y;	// the temporary variables at
int i,j,k;				// loop counters
long int N; // rows, columns of image
char *im_input;		  // pointer to input image
char *im_output;		  // pointer to output image
double start_time, end_time;

// d_x is the gradient in x direction
// d_y is the gradient in y direction

int thread_count[15] = {2,4,8,32,64,128,120,240,16,40,60,200,80,58,100};

char d_x[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};
char d_y[3][3] = {{1,0,-1},{2,0,-2},{1,0,-1}};

//define timer function
double timerval ()
{
	struct timeval st;
	gettimeofday (&st, NULL);
	return st. tv_sec + st. tv_usec * 1e-6;
}
//void sobelSerial(int *im_input, int *im_output, int imSize, int d_x[3][3], int d_y[3][3])
void sobelPara()
{
#pragma omp parallel shared(im_input, im_output) private(i,j,d_x,d_y)
	// the first row remains as it is
#pragma omp for
// v4 specifics
	for(i=0;i<N; i++)
	{	
		im_output[i] = im_input[i];	//first row
		im_output[N*i] = im_input[N*i];	// first column
		im_output[(N-1)*(N) + i] = im_input[N*(N-1) + i];	//last row
		im_output[(N-1)*i] = im_input[(N-1)*i];  //last column
	}	
/*
#pragma omp for
	// the first column remains as it is
	for(i=1; i<N; i++)
		im_output[N*i] = im_input[N*i];
*/
#pragma omp for
	// calculating the rest of the output
	for(i=1; i<N-1; i++)
	{
		for(j=1; j<N-1; j++)
		{

#pragma simd
/*		// the gradient in x direction
			sobel_x = (double)( d_x[0][0]*(im_input[(i-1)*N + j-1]) + d_x[1][0]*(im_input[j-1 +i*N]) + d_x[0][2]*(im_input[j-1 +N*(i+1)]) + d_x[1][0]*(im_input[j + N*(i-1)]) + d_x[1][1]*(im_input[j + N*i]) + d_x[1][2]*(im_input[j +N*(i+1)]) +  d_x[2][0]*(im_input[j+1+N*(i-1)]) + d_x[2][1]*(im_input[j+1 + N*i]) + d_x[2][2]*(im_input[j+1 + N*(i+1)]));
			// the gradient in x direction
			sobel_y = (double) (d_y[0][0]*(im_input[j-1 + N*(i-1)]) + d_y[0][1]*(im_input[j-1 +N*i]) + d_y[0][2]*(im_input[j-1 +N*(i+1)]) + d_y[1][0]*(im_input[j + N*(i-1)]) + d_y[1][1]*(im_input[j +N*i]) + d_y[1][2]*(im_input[j +N*(i+1)]) +  d_y[2][0]*(im_input[j+1 +N*(i-1)]) + d_y[2][1]*(im_input[j+1 +N*i]) + d_y[2][2]*(im_input[j+1 + N*(i+1)]));
*/
	// the gradient in x direction
			sobel_x = (double)( 1*(im_input[(i-1)*N + j-1]) + 0*(im_input[j-1 +i*N]) - 1*(im_input[j-1 +N*(i+1)]) + 2*(im_input[j + N*(i-1)]) + 0*(im_input[j + N*i]) -2*(im_input[j +N*(i+1)]) +1*(im_input[j+1+N*(i-1)]) +0*(im_input[j+1 + N*i]) -1*(im_input[j+1 + N*(i+1)]));
			// the gradient in x direction
			sobel_y = (double) (1*(im_input[j-1 + N*(i-1)]) + 2*(im_input[j-1 +N*i]) -1*(im_input[j-1 +N*(i+1)]) + 0*(im_input[j + N*(i-1)]) +0*(im_input[j +N*i]) + 0*(im_input[j +N*(i+1)]) -1*(im_input[j+1 +N*(i-1)]) -2*(im_input[j+1 +N*i]) -1*(im_input[j+1 + N*(i+1)]));

			// calculating the absolute value of pixel from from the x and y filter

			im_output[j + N*i] = (char) sqrt(pow(sobel_x,2) + pow(sobel_y,2));

			//printf("%d the value of output array is %d \n", (i*N+j),im_output[i*N+j]);

			// truncating values beyond 255 for grayscale images
			if (im_output[j+N*i] > 255)
			{
				(im_output[j+N*i]) = 255;
			}
		}
//	np = omp_get_num_threads();
	}
//	#pragma omp barrier
}

/**
 * Initializing the image matrix
 */
void initializeImage()
{
	for(i=0; i<N*N; i++)
	{
		im_input[i] = (char) (rand()%255);
		//printf("%d the value of input array is %d \n", (i),im_input[i]);
	}
}

int main(int argc, char* argv[])
{
	int m =0;
	int l =0;
	int kStart =2;
	int kStop = 11;
	double seconds[kStop];

	for(l = 3; l<5;l++)
	{
	for(k = kStart; k<= kStop; k++)
	{
		omp_set_num_threads(thread_count[l]);
//		omp_set_num_threads(l);
		omp_set_dynamic(0);
		N = pow(2,k);
		// memory allocation for the input and output images
//		im_input  = (char *)mkl_malloc(sizeof(char)*N*N,64);
//		im_output = (char *)mkl_malloc(sizeof(char)*N*N,64);
		im_input  = (char *)malloc(sizeof(char)*N*N);
		im_output = (char *)malloc(sizeof(char)*N*N);


		initializeImage();
		start_time = timerval();
		for(i=0;i<1000;i++)
		{
			sobelPara(); //(im_input, im_output, N, d_x, d_y);
		}

		end_time = timerval();
		seconds[m++] = (end_time-start_time)/1000;

		//printf("Time taken for N = %ld is %f\n", N, seconds);


		//fprintf(output1,"Time taken for N = %ld is %lf\n", N, (end_time-start_time)/1000);
		if(im_input != NULL && im_output!= NULL)
		{
			//mkl_free(im_input);
			//mkl_free(im_output);
			free(im_input);
			free(im_output);

		}

	}	// k loop finishes
	
	// printing to file
	FILE *sobelParallelOutputfile;
	sobelParallelOutputfile = fopen("sobelParallelBenchmarking.csv","a+");
	if(sobelParallelOutputfile == NULL)
	{
		printf("Could not open file\n");
		return EXIT_FAILURE;
	}
	fprintf(sobelParallelOutputfile,"N, Time taken, thread_count = %d \n", thread_count[l]);
	for(i=0;i<m;i++)
	{
	fprintf(sobelParallelOutputfile,"%lf,%f\n", pow(2,(i+kStart)), seconds[i]);
	}
	fclose(sobelParallelOutputfile); //Closing the file
	m=0;
	}	// l loop finishes
	return EXIT_SUCCESS;
}
