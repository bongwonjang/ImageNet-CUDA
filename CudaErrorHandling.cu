#include "CudaErrorHandling.h"

/*
	- checkXXX function
		The 3 functions, checkCUDA, checkCUDNN, and checkCUBLAS, verify runtime errors.
		
	- checkXXX 함수
		checkCUDA, checkCUDNN, checkCUBLAS 3가지 함수는 런타임 에러를 확인해준다.
*/
void checkCUDA(cudaError_t status, const char * file, int line)
{
	if (status != CUDA_SUCCESS)
	{
		printf("%s %s %d\n", cudaGetErrorString(status), file, line);
		exit(-1);
	}
}

void checkCUDNN(cudnnStatus_t status, const char * file, int line)
{
	if (status != CUDNN_STATUS_SUCCESS)
	{
		printf("%s %s %d\n", cudnnGetErrorString(status), file, line);
		exit(-1);
	}
}

void checkCUBLAS(cublasStatus_t status, const char * file, int line)
{
	if (status != CUBLAS_STATUS_SUCCESS)
	{
		printf("%s %s %d\n", cublasGetErrorString(status), file, line);
		exit(-1);
	}
}

/*
	- CudaContext::CudaContext
		Create cublas and cudnn handlers for using cuBLAS and cuDNN APIs.
		We will name these handlers the 'CUDA context'.

	- CudaContext::CudaContext
		cuBLAS, cuDNN API를 사용하기 위한 cublas, cudnn handler를 생성.
		이러한 handler들을 'CUDA context'라고 이름을 붙이겠다.
*/
CudaContext::CudaContext()
{
	cublasCreate(&cublas_handle_);
	checkCUDNN(cudnnCreate(&cudnn_handle_), __FILE__, __LINE__);
}
CudaContext::~CudaContext()
{
	cublasDestroy(cublas_handle_);
	checkCUDNN(cudnnDestroy(cudnn_handle_), __FILE__, __LINE__);
}
