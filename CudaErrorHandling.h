#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include <iostream>
#include <cstdlib>

/*
	- cublasGetErrorString function
		During the code development, there was no `XXXGetErrorString` function available for the cuBLAS API.
		Therefore, we created a custom function to analyze cuBLAS errors.
		
	- cublasGetErrorString 함수
		코드 작성 과정에서 cuBLAS API에 대한 `XXXGetErrorString` 함수가 없었다.
		따라서, 자체적으로 cuBLAS 에러를 분석하는 함수를 제작.
*/
static const char *cublasGetErrorString(cublasStatus_t error)
{
	switch (error)
	{
	case CUBLAS_STATUS_SUCCESS:
		return "CUBLAS_STATUS_SUCCESS";

	case CUBLAS_STATUS_NOT_INITIALIZED:
		return "CUBLAS_STATUS_NOT_INITIALIZED";

	case CUBLAS_STATUS_ALLOC_FAILED:
		return "CUBLAS_STATUS_ALLOC_FAILED";

	case CUBLAS_STATUS_INVALID_VALUE:
		return "CUBLAS_STATUS_INVALID_VALUE";

	case CUBLAS_STATUS_ARCH_MISMATCH:
		return "CUBLAS_STATUS_ARCH_MISMATCH";

	case CUBLAS_STATUS_MAPPING_ERROR:
		return "CUBLAS_STATUS_MAPPING_ERROR";

	case CUBLAS_STATUS_EXECUTION_FAILED:
		return "CUBLAS_STATUS_EXECUTION_FAILED";

	case CUBLAS_STATUS_INTERNAL_ERROR:
		return "CUBLAS_STATUS_INTERNAL_ERROR";
	}

	return "<unknown>";
}

/*
	- checkXXX function
		The 3 functions, checkCUDA, checkCUDNN, and checkCUBLAS, verify runtime errors.
		
	- checkXXX 함수
		checkCUDA, checkCUDNN, checkCUBLAS 3가지 함수는 런타임 에러를 확인해준다.
*/
void checkCUDNN(cudnnStatus_t status, const char * file, int line);
void checkCUDA(cudaError_t status, const char * file, int line);
void checkCUBLAS(cublasStatus_t status, const char * file, int line);

/*
	- CudaContext class
		A class representing handlers for cuBLAS and cuDNN APIs.
		
	- CudaContext 클래스
		cuBLAS, cuDNN API를 사용하는 handler를 대표하는 클래스.
*/
class CudaContext
{
public:
	CudaContext();
	~CudaContext();

	cublasHandle_t getCublasHandle()
	{
		return cublas_handle_;
	}

	cudnnHandle_t getCudnnHandle()
	{
		return cudnn_handle_;
	}

	const float one = 1.0f;
	const float zero = 0.0f;
	const float minus_one = -1.0f;

private:
	cublasHandle_t cublas_handle_;
	cudnnHandle_t cudnn_handle_;
};
