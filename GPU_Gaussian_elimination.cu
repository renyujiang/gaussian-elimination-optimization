///////////code can run

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>


typedef double data_t;

#define L 1024
#define W 1024
#define error 0.05






void initializeArray(data_t** array, int seed)
{
	srand(seed);
	int i, j;
	if (seed == 0)
	{
		for (i = 0; i < L; i++)
		{
			for (j = 0; j < W + 1; j++)
			{
				array[i][j] = 0;
			}
		}
		return;
	}

	for (i = 0; i < L; i++)
	{
		for (j = 0; j < W + 1; j++)
		{
			int sign = rand() % 2 ? -1 : 1;
			double rand_float = (float)rand() / (float)RAND_MAX * sign;
			array[i][j] = rand_float;
		}
	}
	return;
}




//gaussian_elimination_kernel
__global__ void pivot_and_swap(data_t* a, int n, int k) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n) return;

	if (i == k) {
		int maxIndex = i;
		data_t maxValue = fabs(a[i * (n + 1) + k]);

		// find the largest
		for (int j = i + 1; j < n; j++) {
			data_t currentValue = fabs(a[j * (n + 1) + k]);
			if (currentValue > maxValue) {
				maxIndex = j;
				maxValue = currentValue;
			}
		}

		// swap lines
		if (maxIndex != i) {
			for (int j = k; j <= n; j++) {
				data_t temp = a[i * (n + 1) + j];
				a[i * (n + 1) + j] = a[maxIndex * (n + 1) + j];
				a[maxIndex * (n + 1) + j] = temp;
			}
		}
	}
}

__global__ void elimination(data_t* a, int n, int k) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n) return;

	if (i > k) {
		data_t factor = a[i * (n + 1) + k] / a[k * (n + 1) + k];
		for (int j = k + 1; j <= n; j++) {
			a[i * (n + 1) + j] -= factor * a[k * (n + 1) + j];
		}
	}
}



data_t* gaussian_elimination(data_t* a, int n) {
	data_t* d_a;
	int size = n * (n + 1) * sizeof(data_t);
	cudaMalloc((void**)&d_a, size);
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

	dim3 blockSize(64);
	dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

	for (int k = 0; k < n - 1; k++) {
		pivot_and_swap << <gridSize, blockSize >> > (d_a, n, k);
		cudaDeviceSynchronize();
		elimination << <gridSize, blockSize >> > (d_a, n, k);
		cudaDeviceSynchronize();
	}
	cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);
	cudaFree(d_a);

	data_t* x = (data_t*)malloc(n * sizeof(data_t));
	x[n - 1] = a[n * (n + 1) - 1] / a[n * (n + 1) - 2];
	for (int i = n - 2; i >= 0; i--) {
		data_t s = 0;
		for (int j = i + 1; j < n; j++) {
			s += a[i * (n + 1) + j] * x[j];
		}
		x[i] = (a[i * (n + 1) + n] - s) / a[i * (n + 1) + i];
	}
	return x;

}


//Base Cpu
data_t* gaussian_elimination_base(data_t** a, int n) {
	data_t s, p;
	int i, j, k;
	data_t* x = (data_t*)malloc((L) * sizeof(data_t));
	for (k = 0; k <= n - 1; k++)
	{
		for (i = k + 1; i < n; i++)
		{
			p = a[i][k] / a[k][k];
			for (j = k; j <= n; j++)
			{
				a[i][j] = a[i][j] - (p * a[k][j]);
			}
		}
	}
	x[n - 1] = a[n - 1][n] / a[n - 1][n - 1];
	for (i = n - 2; i >= 0; i--)
	{
		s = 0;
		for (j = i + 1; j < n; j++)
		{
			s += (a[i][j] * x[j]);
		}
		x[i] = (a[i][n] - s) / a[i][i]; // Move this line outside the inner loop
	}
	return x;
}


//ckeck input matrix
void print_matrix(data_t** a, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			printf("%f\t", a[i][j]);
		}
		printf("\n");
	}
}


data_t check_result(data_t* matrix1, data_t* matrix2) {

	//int len1 = sizeof(matrix1) / sizeof(matrix1[0]);
	//int len2 = sizeof(matrix2) / sizeof(matrix2[0]);
	int len1 = L;
	int len2 = L;
	if (len1 != len2) {
		printf("Unable to compare, the two matrixes are not the same size");
		return 0;
	}
	int i;
	data_t real_error, max_error = 0.0;
	for (i = 0; i < len1; i++) {
		real_error = fabs(matrix1[i] - matrix2[i]) / matrix1[i];
		if (real_error > error) {
			printf("The two matrixes' error is larger than the error rate");
			return real_error;
		}
		if (real_error > max_error) {
			max_error = real_error;
		}
	}
	return max_error;
}




int main() {
	data_t** a = (data_t**)
		malloc(L * sizeof(data_t*));
	for (int i = 0; i < L; i++) {
		a[i] = (data_t*)malloc((W + 1) * sizeof(data_t));
	}

	initializeArray(a, 1);


	//put a into 1-D array linearized_a
	data_t* linearized_a = (data_t*)malloc(L * (W + 1) * sizeof(data_t));
	for (int i = 0; i < L; i++) {
		for (int j = 0; j < W + 1; j++) {
			linearized_a[i * (W + 1) + j] = a[i][j];
		}
	}

	//put a into 2-D array a_base
	data_t** a_base = (data_t**)malloc(L * sizeof(data_t*));
	for (int i = 0; i < L; i++) {
		a_base[i] = (data_t*)malloc((W + 1) * sizeof(data_t));
	}
	for (int i = 0; i < L; i++) {
		for (int j = 0; j < W + 1; j++) {
			a_base[i][j] = a[i][j];
		}
	}


	//Result base
	clock_t start_base = clock();
	data_t* x_base = gaussian_elimination_base(a_base, L);
	clock_t end_base = clock();
	double elapsed_time_base = (double)(end_base - start_base) * 1000.0 / CLOCKS_PER_SEC;


	//Result in GPU
	clock_t start = clock();
	data_t* x = gaussian_elimination(linearized_a, L);
	clock_t end = clock();
	double elapsed_time = (double)(end - start) * 1000.0 / CLOCKS_PER_SEC;

	// Compare results and calculate max error
	data_t max_error = check_result(x_base, x);

	// Output the results 

	//print_matrix(a, L, W + 1);
	printf("Base Result:\n");
	for (int i = 0; i < L; i++) {
		printf("x_base[%d] = %f\n", i, x_base[i]);
	}


	printf("Result:\n");
	for (int i = 0; i < L; i++) {
		printf("x[%d] = %f\n", i, x[i]);
	}

	printf("\nBase Elapsed Time: %f ms\n", elapsed_time_base);
	printf("\nElapsed Time: %f ms\n", elapsed_time);
	printf("\nMax Error: %f\n", max_error);
	// Free the allocated memory
	for (int i = 0; i < L; i++) {
		free(a[i]);
	}


	free(a);
	free(linearized_a);
	free(x);

	return 0;
}