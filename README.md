# gaussian-elimination-optimization
### Group Member: Renyu Jiang, Yimin Xu, Mingda Li
## Report Link:
https://docs.google.com/document/d/1rLByw_EIiKQAqwEFzTsJNqccYC1G9NsrCUy_pFIfb4Q/edit?usp=sharing

## Description of Gaussian_Elimination_Base.c
This is a C program that implements the Gaussian elimination method to solve a system of linear equations. The program initializes a 2D array of size L x (W+1) with random numbers within the range of -1 to 1, where L and W are pre-defined constants. The program then performs Gaussian elimination on the array to solve the linear system, and prints the solution. The program also calculates the time it takes to perform the calculations using the CLOCK_REALTIME clock, and prints it at the end.

The program uses a single thread to perform the calculations. The "gaussian_elimination_base" function performs the Gaussian elimination on the given 2D array and returns the solution in the form of an array of data_t type. The "check_result" function is used to compare the calculated solution with the actual solution of the linear system and returns the maximum error between the two solutions.
