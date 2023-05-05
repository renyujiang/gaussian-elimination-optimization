# gaussian-elimination-optimization
### Group Member: Renyu Jiang, Yimin Xu, Mingda Li

## Description

Gaussian elimination is a widely used algorithm in linear algebra that helps to solve systems of linear equations. It is a step-by-step procedure that reduces a system of linear equations to an equivalent system that is easier to solve. The algorithm works by performing a sequence of computations on an augmented matrix, which is a matrix that includes both the coefficients of the variables and the constants of the equations. The goal of Gaussian elimination is to transform the augmented matrix into row echelon form, which means that the leading coefficient (the first non-zero entry) of each row is to the right of the leading coefficient of the row above it. After transforming the matrix into row echelon form, back-substitution can be used to solve these equations.

To optimize gaussian elimination, we propose three optimization ideas: AVX instruction optimization, CPU multithreading optimization and NVIDIA GPU Cuda optimization. 

## Compilation and Running Instructions
The shell codes to compile and run these codes are not complicated, basically gcc instructions plus some parameters.
AVX optimization
To compile and run avx.c, use the following command:

gcc -O1 -mavx -std=gnu99 avx.c -lm -lrt -o avx
./avx

Multi-thread optimization and AVX optimization in threads
Multi-thread with AVX optimization code is in file multi_threads.c, to compile and run it, use following code: 

gcc -mavx2 -pthread -O1 -o multi_threads multi_threads.c
./multi_threads

There is also a version of code which has multi-thread optimization in file multi_threads_original.c, to compile and run it, use following code: 

gcc -pthread -O1 -o multi_threads_original multi_threads_original.c
./multi_threads_original
GPU Cuda optimization
All cuda related code is in the file GPU_Gaussian_elimination.cu. This is a separately completed file including baseline code. 
To run this code in linux, you need use following code:

```shell
 module load cuda/ 10.0 
-arch compute_70 -code sm_70 GPU_Gaussian_elimination.cu -o GPU_Gaussian_elimination
./GPU_Gaussian_elimination
```

Make sure to use corresponding compile parameters.

## File list
baseline.c
avx.c
multi_threads.c
multi_threads_original.c
GPU_Gaussian_elimination.cu

