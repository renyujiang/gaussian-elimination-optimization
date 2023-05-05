
/*
    this code is modified from baseline.c
    this code is the base code for Gaussian Elimination Optimization
    to compile this code, use the following command:
    gcc -O1 -mavx -std=gnu99 avx.c -lm -lrt -o avx
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <omp.h>
#include <xmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

#define OPTIONS 10
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
void initializeArray(data_t **array, int seed,int n)
{
    srand(seed);
    int i, j;
    if (seed == 0)
    {
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < n + 1; j++)
            {
                array[i][j] = 0;
            }
        }
        return;
    }

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n + 1; j++)
        {
            int sign = rand() % 2 ? -1 : 1;
            data_t rand_data_t = (data_t)rand() / (data_t)RAND_MAX * sign;
            array[i][j] = rand_data_t;
        }
    }
    return;
}

data_t *gaussian_elimination_base(data_t **a, int n){
    data_t s,p;
    int i, j, k;
    data_t *x = (data_t *)malloc((n) * sizeof(data_t));
    for (k = 0; k <= n - 1; k++){
        for (i = k + 1 ; i < n; i++){
            p = a[i][k] / a[k][k];
            for (j = k; j <= n; j++) {
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
            x[i] = (a[i][n] - s) / a[i][i];
        }
    }
    return x;
}

data_t *gaussian_elimination_avx(data_t **a, int n){
    int i, j, k,l;
    data_t p,s1,t1;
    data_t *x = (data_t *)malloc((n) * sizeof(data_t));
    __m256 pv,ak,ai;
    for (k = 0; k <= n-1 ; k++){
        for (i = k + 1 ; i < n; i++){
            p = a[i][k] / a[k][k];
            pv = _mm256_broadcast_ss(&p);
            for (j = k ; j <= n - 8; j += 8)  {
                ak = _mm256_loadu_ps(&a[k][j]);
                ai = _mm256_loadu_ps(&a[i][j]);
                ai = _mm256_sub_ps(ai, _mm256_mul_ps(pv, ak));
                _mm256_storeu_ps(&a[i][j], ai);
            }
            for (; j <= n ; j ++)  {
                a[i][j] = a[i][j] - (p * a[k][j]);
            }
        }
    }
    x[n - 1] = a[n - 1][n] / a[n - 1][n - 1];
    for (i = n - 2; i >= 0; i--)
    {
        s1 = 0;
        for (j = i + 1; j < n; j++)
        {
            s1 += (a[i][j] * x[j]);
            x[i] = (a[i][n] - s1) / a[i][i];
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
    int L[10] = {256,512,1000,2000,3000,4096,5000,6000,7000,8192};
    int W[10] = {256,512,1000,2000,3000,4096,5000,6000,7000,8192};
    double time_stamp[OPTIONS][3];
    int size[OPTIONS];
    for (int OPTION = 0; OPTION<OPTIONS; OPTION++) {
        double s, p;
        int i, j, k, n;
        n = W[OPTION];
        size[OPTION] =  L[OPTION];
        data_t **a = (data_t **)malloc(n * sizeof(data_t *));
        for (i = 0; i < n; i++)
        {
            a[i] = (data_t *)malloc((n + 1) * sizeof(data_t));
        }
        initializeArray(a, 1,n);
        data_t cpu_time;
        struct timespec time_start, time_stop;
        clock_gettime(CLOCK_REALTIME, &time_start);
        data_t *x = gaussian_elimination_base(a, n);
        clock_gettime(CLOCK_REALTIME, &time_stop);
        time_stamp[OPTION][0] = interval(time_start, time_stop);
        for ( i = 0; i < n; i++)
        {
            free(a[i]);
        }
        free(a);
        free(x);
        a = (data_t **)malloc(n * sizeof(data_t *));
        for (i = 0; i < n; i++)
        {
            a[i] = (data_t *)malloc((n + 1) * sizeof(data_t));
        }
        initializeArray(a, 1,n);
        clock_gettime(CLOCK_REALTIME, &time_start);
        data_t *x1 = gaussian_elimination_avx(a, n);
        clock_gettime(CLOCK_REALTIME, &time_stop);
        time_stamp[OPTION][1] = interval(time_start, time_stop);
    }
    printf("\nAll times are in seconds\n");
    printf("size, base, avx\n");
    {
        int i, j;
        for (i = 0; i < OPTIONS; i++) {
            printf("%4ld", size[i]);
            for (j = 0; j < 2; j++) {
                printf(",%10.4g", time_stamp[i][j]);
            }
            printf("\n");
        }
    }

    return 0;
}