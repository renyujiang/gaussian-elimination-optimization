/*
    this code is modified from baseline.c
    this code is the multi-thread code for Gaussian Elimination Optimization
    to compile this code, use the following command:
    gcc -mavx2 -pthread -O1 -o multi_threads multi_threads.c
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include <immintrin.h>

#define L 8000
#define W L

#define data_t float

#define NUM_THREADS 4

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
    for (k = 0; k < n; k++)
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
            x[i] = (a[i][n] - s) / a[i][i];
        }
    }
    return x;
}

typedef struct {
    int start;
    int end;
    int k;
    int n;
    data_t **a;
} thread_data_t;

pthread_barrier_t barrier;

void *gaussian_elimination_thread(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    int row, col;
    data_t pivot_ratio;
    __m256 pivot_ratio_vec, row_vec, pivot_vec, result_vec;

    for (row = data->start; row < data->end; row++) {
        pivot_ratio = data->a[row][data->k] / data->a[data->k][data->k];
        pivot_ratio_vec = _mm256_set1_ps(pivot_ratio);

        for (col = data->k; col <= data->n - 7; col += 8) {
            row_vec = _mm256_loadu_ps(&(data->a[row][col]));
            pivot_vec = _mm256_loadu_ps(&(data->a[data->k][col]));
            result_vec = _mm256_sub_ps(row_vec, _mm256_mul_ps(pivot_ratio_vec, pivot_vec));
            _mm256_storeu_ps(&(data->a[row][col]), result_vec);
        }

        // Handle the remaining loop iterations
        for (; col <= data->n; col++) {
            data->a[row][col] -= pivot_ratio * data->a[data->k][col];
        }
    }

    pthread_barrier_wait(&barrier);

    return NULL;
}

data_t *gaussian_elimination_multi_threads(data_t **a, int n) {
    data_t s;
    int i, j, k;
    data_t *x = (data_t *)malloc(n * sizeof(data_t));
    pthread_t threads[NUM_THREADS];
    thread_data_t thread_data[NUM_THREADS];
    int chunk_size;

    pthread_barrier_init(&barrier, NULL, NUM_THREADS);

    for (k = 0; k < n; k++) {
        chunk_size = (n - k - 1) / NUM_THREADS;

        for (i = 0; i < NUM_THREADS; i++) {
            thread_data[i].start = k + 1 + i * chunk_size;
            thread_data[i].end = i == 3 ? n : k + 1 + (i + 1) * chunk_size;
            thread_data[i].k = k;
            thread_data[i].n = n;
            thread_data[i].a = a;
            pthread_create(&threads[i], NULL, gaussian_elimination_thread, &thread_data[i]);
        }

        for (i = 0; i < NUM_THREADS; i++) {
            pthread_join(threads[i], NULL);
        }
    }

    pthread_barrier_destroy(&barrier);

    x[n - 1] = a[n - 1][n] / a[n - 1][n - 1];
    int row, col;

for (row = n - 2; row >= 0; row--) {
    float sum = 0;

    for (col = row + 1; col < n; col++) {
        sum += a[row][col] * x[col];
    }

    x[row] = (a[row][n] - sum) / a[row][row];
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

    data_t *x = gaussian_elimination_multi_threads(a, n);

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
