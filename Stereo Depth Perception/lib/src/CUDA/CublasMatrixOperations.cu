#include "Wrapper.h"

VOID MatrixMultiplication_GPU(
	cublasHandle_t& handle,
	CONST IN float* A,
	CONST IN float* B,
	CONST IN int m,
	CONST IN int k,
	CONST IN int n,
	OUT float* C)
{
	int lda = m, ldb = k, ldc = m;
	const float alpha = 0.0f;
	const float beta = 1.0f;

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 
		&alpha, A, lda, B, ldb, &beta, C, ldc);
}

VOID ScalarMatrixMultiplication_GPU(
	cublasHandle_t& handle,
	CONST IN float alpha,
	CONST IN int m,
	CONST IN int n,
	OUT float* A)
{
	cublasSscal(handle, m * n, &alpha, A, 1);
}

VOID MatrixAddition_GPU(
	cublasHandle_t& handle,
	CONST IN float* A,
	CONST IN int m,
	CONST IN int n,
	OUT float* B)
{
	const float alpha = 1.0f;

	cublasSaxpy(handle, m * n, &alpha, A, 1, B, 1);
}

VOID CUBLAS::MatrixMultiplication(
	CONST IN float* A,
	CONST IN float* B,
	CONST IN int m,
	CONST IN int k,
	CONST IN int n,
	OUT float* C)
{
	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	MatrixMultiplication_GPU(handle, A, B, m, k, n, C);

	// Destroy the handle
	cublasDestroy(handle);
}

VOID CUBLAS::ScalarMatrixMultiplication(
	CONST IN float alpha,
	CONST IN int m,
	CONST IN int n,
	OUT float* A)
{
	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	ScalarMatrixMultiplication_GPU(handle, alpha, m, n, A);

	// Destroy the handle
	cublasDestroy(handle);
}

VOID CUBLAS::MatrixAddition(
	CONST IN float* A,
	CONST IN int m,
	CONST IN int n,
	OUT float* B)
{
	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	MatrixAddition_GPU(handle, A, m, n, B);

	// Destroy the handle
	cublasDestroy(handle);
}