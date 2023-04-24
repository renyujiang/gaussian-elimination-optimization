/*
    this code is modified from baseline.c
    this code is the base code for Gaussian Elimination Optimization
    to compile this code, use the following command:
    gcc -O1 -o GE_baseline GE_baseline.c
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <pthread.h>

#define L 4096
#define W 2048

#define data_t float

// this is the error rate
#define error 0.05

double interval(struct timespec start, struct timespec end)
{
    struct timespec temp;
    temp.tv_sec = end.tv_sec - start.tv_sec;
    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    if (temp.tv_nsec < 0)
    {
        temp.tv_sec = temp.tv_sec - 1;
        temp.tv_nsec = temp.tv_nsec + 1000000000;
    }
    return (((double)temp.tv_sec) + ((double)temp.tv_nsec) * 1.0e-9);
}

// Initialize the array with random numbers within -1 to 1
// The first parameter is the array to be initialized, the second parameter is the seed for random number generator
void initializeArray(data_t **array, int seed)
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
            float rand_float = (float)rand() / (float)RAND_MAX * sign;
            array[i][j] = rand_float;
        }
    }
    return;
}

data_t *gaussian_elimination_base(data_t **a, int n){
    data_t s, p;
    int i, j, k;
    data_t *x = (data_t *)malloc((L) * sizeof(data_t));
    for (k = 0; k <= n - 1; k++)
    {
        for (i = k + 1; i < n; i++)
        {
            p = a[i][k] / a[k][k];
            for (j = k; j <= n; j++)
            {
                a[i][j] = a[i][j] - (p * a[k][j]);
                // printf("\n a[%d][%d] = %f", i, j, a[i][j]);
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
            x[i] = (a[i][n] - s) / a[i][i];
        }
    }
    return x;
}

data_t check_result(data_t *matrix1, data_t *matrix2){
    int len1=sizeof(matrix1)/sizeof(matrix1[0]);
    int len2=sizeof(matrix2)/sizeof(matrix2[0]);
    if(len1!=len2){
        printf("Unable to compare, the two matrixes are not the same size");
        return 0;
    }
    int i;
    data_t real_error,max_error;
    for(i=0;i<len1;i++){
        real_error=fabs(matrix1[i]-matrix2[i])/matrix1[i];
        if(real_error>error){
            printf("The two matrixes' error is larger than the error rate");
            return real_error;
        }
        if(real_error>max_error){
            max_error=real_error;
        }
    }
    return max_error;
}

int main()
{
    double s, p;
    int i, j, k, n;
    n = W;

    data_t **a = (data_t **)malloc(L * sizeof(double *));
    for (i = 0; i < L; i++)
    {
        a[i] = (data_t *)malloc((W + 1) * sizeof(double));
    }
    initializeArray(a, 1);


    float cpu_time;
    struct timespec time_start, time_stop;


// start test code on CPU
    clock_gettime(CLOCK_REALTIME, &time_start);

    data_t *x = gaussian_elimination_base(a, n);

    clock_gettime(CLOCK_REALTIME, &time_stop);
    // end test code on CPU

    cpu_time = interval(time_start, time_stop);

    printf("\nThe result is :\n");
    for (i = 0; i < n; i++)
    {
        printf("\nx[%d] = %.2f", i + 1, x[i]);
    }



    printf("\n\nTime used: %f seconds\n", cpu_time);

    printf("\n\n");

    return 0;
}
