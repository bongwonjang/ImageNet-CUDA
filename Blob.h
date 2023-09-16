#pragma once

#include <array>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>

/*
	- Blob class
		A class for managing 'Tensor' in deep learning training.
		It provieds functions to easily handle the Tensor movement between CPU and GPU memory.
		
	- Blob 클래스
		딥러닝 훈련에서 텐서(Tensor)를 관리하는 클래스.
		텐서의 CPU/GPU 메모리 간 이동을 함수로 쉽게 관리할 수 있게 한다.
*/
template <typename ftype>
class Blob
{
public:
	Blob(int n = 1, int c = 1, int h = 1, int w = 1) : n_(n), c_(c), h_(h), w_(w)
	{
		h_ptr_ = new ftype[n_ * c_ * h_ * w_];
	}
	Blob(std::array<int, 4> size) : n_(size[0]), c_(size[1]), h_(size[2]), w_(size[3])
	{
		h_ptr_ = new ftype[n_ * c_ * h_ * w_];
	}
	~Blob()
	{
		if (h_ptr_ != nullptr)
		{
			delete[] h_ptr_;
			if (d_ptr_ != nullptr)
				cudaFree(d_ptr_);
		}

		if (is_tensor_)
			cudnnDestroyTensorDescriptor(tensor_desc_);
	}

	void reset(int n = 1, int c= 1, int h = 1, int w = 1)
	{
		// update size information
		n_ = n;
		c_ = c;
		h_ = h;
		w_ = w;

		// terminate current buffers
		if (h_ptr_ != nullptr)
		{
			delete[] h_ptr_;
			h_ptr_ = nullptr;
		}
		if (d_ptr_ != nullptr)
		{
			cudaFree(d_ptr_);
			d_ptr_ = nullptr;
		}

		// create new buffers
		h_ptr_ = new ftype[n_ * c_ * h_ * w_];
		cuda();

		// reset tensor descriptor if it was tensor
		if (is_tensor_)
		{
			cudnnDestroyTensorDescriptor(tensor_desc_);
			is_tensor_ = false;
		}
	}
	void reset(std::array<int, 4> size)
	{
		reset(size[0], size[1], size[2], size[3]);
	}

	// returns array of tensor shape
	std::array<int, 4> shape() 
	{ 
		return std::array<int, 4>({ n_, c_, h_, w_ }); 
	};

	// returns number of elements for 1 batch
	int size()
	{
		return c_ * h_ * w_;
	}

	// returns number of total elements in blob including batch
	int len()
	{
		return n_ * c_ * h_ * w_;
	}

	// returns size of allocated memory
	int buf_size()
	{
		return sizeof(ftype) * len();
	}

	int getN() { return n_; }
	int getC() { return c_; }
	int getH() { return h_; }
	int getW() { return w_; }

	/* Tensor Control */
	bool is_tensor_ = false;
	cudnnTensorDescriptor_t tensor_desc_;
	cudnnTensorDescriptor_t tensor()
	{
		if (is_tensor_)
			return tensor_desc_;

		cudnnCreateTensorDescriptor(&tensor_desc_);
		if (std::is_same<ftype, float>::value)
			cudnnSetTensor4dDescriptor(tensor_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_, c_, h_, w_);
		else if(std::is_same<ftype, double>::value)
			cudnnSetTensor4dDescriptor(tensor_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, n_, c_, h_, w_);
		else
		{
			std::cout << "Type Error in Blob" << std::endl;
			exit(-1);
		}

		is_tensor_ = true;

		return tensor_desc_;
	}

	/* Memory Control */
	// get specified memory pointer
	ftype * ptr()
	{
		return h_ptr_;
	}

	// get cuda memory
	float * cuda()
	{
		if (d_ptr_ == nullptr)
			cudaMalloc((void**)&d_ptr_, sizeof(ftype) * len());

		return d_ptr_;
	}

	// transfer data between memory
	ftype * to(std::string target)
	{
		ftype * ptr = nullptr;
		if (target == "cuda")	// cuda
		{
			cudaMemcpy(cuda(), h_ptr_, sizeof(ftype) * len(), cudaMemcpyHostToDevice);
			ptr = d_ptr_;
		}
		else					// host
		{
			cudaMemcpy(h_ptr_, cuda(), sizeof(ftype) * len(), cudaMemcpyDeviceToHost);
			ptr = h_ptr_;
		}

		return ptr;
	}

	int file_read(std::string filename)
	{
		std::ifstream file(filename, std::ios::in | std::ios::binary);
		if (!file.is_open())
		{
			std::cout << "Failed to Access " << filename << std::endl;
			return -1;
		}

		file.read((char*)h_ptr_, sizeof(float) * (this->len()));
		this->to("cuda");
		file.close();

		return 0;
	}

	int file_write(std::string filename)
	{
		std::ofstream file(filename, std::ios::out | std::ios::binary);
		if (!file.is_open())
		{
			std::cout << "Failed to Write " << filename << std::endl;
			return -1;
		}

		file.write((char*)this->to("host"), sizeof(float) * (this->len()));
		file.close();

		return 0;
	}

private:
	ftype * h_ptr_ = nullptr;
	ftype * d_ptr_ = nullptr;

	int n_ = 1;
	int c_ = 1;
	int h_ = 1;
	int w_ = 1;
};
