/*
	Based on https://github.com/PacktPublishing/Learn-CUDA-Programming,
	we modified some layer classes.
*/
#include <random>
#include <math.h>
#include <algorithm>
#include <assert.h>
#include <sstream>

#include "Layers.h"

__global__ void initKernel(float * d_ptr, float val, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size)
		d_ptr[idx] = val;
}

/********************************************************
* Layer Definition										*
********************************************************/
Layer::Layer()
{
	/* Do nothing */
}

Layer::~Layer()
{
	// TODO: Write a correct destructor.
	//
	//	if (output_ != nullptr)
	//		delete output_;
	//
	//	if (grad_input_ != nullptr)
	//		delete grad_input_;
	//
	//	if (weights_ != nullptr)
	//		delete weights_;
	//
	//	if (biases_ != nullptr)
	//		delete biases_;
	//
	//	if (grad_weights_ != nullptr)
	//		delete grad_weights_;
	//
	//	if (grad_biases_ != nullptr)
	//		delete grad_biases_;
	//
	//	/*
	//		Momentum Algorithm
	//	*/
	//	if (momentum_weights_ != nullptr)
	//		delete momentum_weights_;
	//
	//	if (momentum_biases_ != nullptr)
	//		delete momentum_biases_;
}

void Layer::init_weight_bias(unsigned int seed)
{
	checkCUDA(cudaDeviceSynchronize(), __FILE__, __LINE__);

	if (weights_ == nullptr || biases_ == nullptr)
		return;

	// Create random network
	std::mt19937 gen(2030);

	// He initialize
	//	float range = sqrt(6.0f / input_->size());
	//	std::uniform_real_distribution<float> dis(-range, range);

	// Xavier initialize
	float range = sqrt(3.0f / input_->size());
	std::uniform_real_distribution<float> dis(-range, range);

	for (int i = 0; i < weights_->len(); i++)
		weights_->ptr()[i] = dis(gen);

	for (int i = 0; i < biases_->len(); i++)
		biases_->ptr()[i] = 1.0f;

	// copy initialized value to the device
	weights_->to("cuda");
	biases_->to("cuda");
}
__global__ void momentum_algorithm
				(float * velocity,
				float * gradient,
				float mu, 			/* momentum_term */
				int src_size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < src_size)
	{
		/*
		 * https://pytorch.org/docs/1.4.0/_modules/torch/optim/sgd.html#SGD
		 *
		 * vel = momentum_term * vel + (1 - dampening) * grad
		 * grad = grad + momentum_term * vel
		 */
		float grad = gradient[idx];
		velocity[idx] = mu * velocity[idx] + grad;
		gradient[idx] = mu * velocity[idx] + grad;
	}

}

void Layer::update_weight_bias(float learning_rate)
{
	/* UPDATED : 2021 02 02
	 *
	 * By following PyTorch's torch.optim.sgd source code, weight decay is applied before momentum algorithm.
	 * Also, the applied momentum algorithm is not based on the original paper.
	 * the algorithm is modified in the Pytorch's source code, so I just followed it.
	 *
	 * https://pytorch.org/docs/1.4.0/_modules/torch/optim/sgd.html#SGD
	 */

	// momentum
	float eps = -1.0f * learning_rate;	// -1.0f * learning_rate
	float momentum_term = 0.9f;			// 0.9 is almost default value
	float decay_factor = 0.0001f;

	if (weights_ != nullptr && grad_weights_ != nullptr)
	{
		checkCUBLAS(cublasSaxpy(cuda_->getCublasHandle(),
								grad_weights_->len(),
								&decay_factor,
								weights_->cuda(), // src_x
								1,
								grad_weights_->cuda(), // dst_y <- scale * x + y
								1), __FILE__, __LINE__);

		if(momentum_weights_ != nullptr)
		{
			momentum_algorithm<<<grad_weights_->len() / 1024 + 1, 1024>>>(momentum_weights_->cuda(),
																		grad_weights_->cuda(),
																		momentum_term, momentum_weights_->len());
		}

		/*
			y = y + alpha * x
			W = W + eps * dW
		*/
		checkCUBLAS(cublasSaxpy(cuda_->getCublasHandle(), weights_->len(), &eps, grad_weights_->cuda(), 1, weights_->cuda(), 1), __FILE__, __LINE__);
	}

	if (biases_ != nullptr && grad_biases_ != nullptr)
	{
		checkCUBLAS(cublasSaxpy(cuda_->getCublasHandle(),
								grad_biases_->len(),
								&decay_factor,
								biases_->cuda(),
								1,
								grad_biases_->cuda(),
								1), __FILE__, __LINE__);

		if(momentum_biases_ != nullptr)
		{
			momentum_algorithm<<<grad_biases_->len() / 1024 + 1, 1024>>>(momentum_biases_->cuda(),
																		grad_biases_->cuda(),
																		momentum_term, momentum_biases_->len());
		}

		/*
			b = b + eps * db
		*/
		checkCUBLAS(cublasSaxpy(cuda_->getCublasHandle(), biases_->len(), &eps, grad_biases_->cuda(), 1, biases_->cuda(), 1), __FILE__, __LINE__);
	}
}

float Layer::get_loss(Blob<float> * target)
{
	assert("No Loss layer has no loss" && false);
	return EXIT_FAILURE;
}

float Layer::get_accuracy(Blob<float> * target)
{
	assert("No Loss layer cannot estimate accuracy." && false);
	return EXIT_FAILURE;
}

int Layer::load_parameter()
{
	std::stringstream filename_weights, filename_biases;

	// load weights and biases pretrained parameters
	filename_weights << name_ << ".bin";
	if (weights_->file_read(filename_weights.str()))
		return -1;

	filename_biases << name_ << ".bias.bin";
	if (biases_->file_read(filename_biases.str()))
		return -2;

	std::cout << ".. loaded " << name_ << " pretrained parameter.." << std::endl;

	return 0;
}

int Layer::save_parameter()
{
	std::stringstream filename_weights, filename_biases;

	std::cout << ".. saving " << name_ << " paraemter .." << std::endl;

	// Write weights file
	if (weights_)
	{
		filename_weights << name_ << ".bin";
		if (weights_->file_write(filename_weights.str()))
			return -1;
	}

	// Write bias file
	if (biases_)
	{
		filename_biases << name_ << ".bias.bin";
		if (biases_->file_write(filename_biases.str()))
			return -2;
	}

	std::cout << "save done.." << std::endl;

	return 0;
}

/********************************************************
* Dense Layer (Fully Connected Layer) Definition	*
********************************************************/

Dense::Dense(std::string name, int output_size)
{
	name_ = name;
	output_size_ = output_size;
}

Dense::~Dense()
{
	if (d_one_vec != nullptr)
		cudaFree(d_one_vec);
}
Blob<float> * Dense::forward(Blob<float> * input, bool is_eval)
{
	// initialize weights and biases, if they are nullptr
	if (weights_ == nullptr)
	{
		// setup parameter size information
		input_size_ = input->getC() * input->getH() * input->getW();

		// initialize weight, bias, and output
		weights_ = new Blob<float>(1, 1, input_size_, output_size_);
		biases_ = new Blob<float>(1, 1, output_size_);

		/*
			IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!!
			IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!!
			IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!!

			Momentum Algorithm
		*/
		momentum_weights_ = new Blob<float>(weights_->shape());
		momentum_biases_ = new Blob<float>(biases_->shape());

		cudaMemset(momentum_weights_->cuda(), 0, momentum_weights_->len() * sizeof(float));
		cudaMemset(momentum_biases_->cuda(), 0, momentum_biases_->len() * sizeof(float));
	}

	// initialize input and output
	if (input_ == nullptr || batch_size_ != input->getN())
	{
		input_ = input;
		batch_size_ = input->getN();

		if (output_ == nullptr)
			output_ = new Blob<float>(batch_size_, output_size_);
		else
			output_->reset(batch_size_, output_size_);

		// tensorize
		output_->tensor();

		if (d_one_vec != nullptr)
			cudaFree(d_one_vec);
		checkCUDA(cudaMalloc((void**)&d_one_vec, sizeof(float) * batch_size_), __FILE__, __LINE__);
		initKernel<<< (batch_size_ / 1024) + 1, 1024 >>> (d_one_vec, 1, batch_size_);

		// initialize weights and biasaes
		if (load_pretrain_ && !freeze_)
		{
			if (load_parameter())
			{
				std::cout << "error occured in loading parameter" << std::endl;
				exit(-1);
			}
		}
		else if (!freeze_)
		{
			init_weight_bias();
		}
		else
		{
			/* do nothing */
		}
	}

	// m x n = (m x k) * (k x n)
	// output = weights^T * input											m			n			k
	checkCUBLAS(cublasSgemm(cuda_->getCublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N, output_size_, batch_size_, input_size_,
							&cuda_->one,
							weights_->cuda(), input_size_,
							input->cuda(), input_size_,
							&cuda_->zero,
							output_->cuda(), output_size_), __FILE__, __LINE__);

	// output = output + biases * onevec^T
	checkCUBLAS(cublasSgemm(cuda_->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, output_size_, batch_size_, 1,
							&cuda_->one,
							biases_->cuda(), output_size_,
							d_one_vec, 1,
							&cuda_->one,
							output_->cuda(), output_size_), __FILE__, __LINE__);

	return output_;
}

Blob<float> * Dense::backward(Blob<float> * grad_output, bool is_eval)
{
	if (grad_weights_ == nullptr)
	{
		grad_weights_ = new Blob<float>(weights_->shape());
		grad_biases_ = new Blob<float>(biases_->shape());
	}

	if (grad_input_ == nullptr || batch_size_ != grad_output->getN())
	{
		grad_output_ = grad_output;

		if (grad_input_ == nullptr)
			grad_input_ = new Blob<float>(input_->shape());
		else
			grad_input_->reset(input_->shape());
	}

	/*
		derivation on bias : db = (dy) * d_one_vec

		y = Wx + b
		dy/db = 1
		db = dW
	*/
	checkCUBLAS(cublasSgemv(cuda_->getCublasHandle(), CUBLAS_OP_N, output_size_, batch_size_,
							&cuda_->one,
							grad_output->cuda(), output_size_,
							d_one_vec, 1,
							&cuda_->zero,
							grad_biases_->cuda(), 1), __FILE__, __LINE__);

	/*
		derivation on weight : dW = x * (dy)^T

		y = Wx + b
		dy/dW = x
	*/
	checkCUBLAS(cublasSgemm(cuda_->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_T, input_size_, output_size_, batch_size_,
							&cuda_->one,
							input_->cuda(), input_size_,
							grad_output->cuda(), output_size_,
							&cuda_->zero,
							grad_weights_->cuda(), input_size_), __FILE__, __LINE__);

	/*
		derivation on data : dx = W * dy

		y = Wx + b
		dy/dx = W
	*/
	if (!gradient_stop_)
	{
		checkCUBLAS(cublasSgemm(cuda_->getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N,
								input_size_, batch_size_, output_size_,
								&cuda_->one,
								weights_->cuda(), input_size_,
								grad_output->cuda(), output_size_,
								&cuda_->zero,
								grad_input_->cuda(), input_size_), __FILE__, __LINE__);
	}

	return grad_input_;
}

/********************************************************
* Activation Layer Definition							*
********************************************************/
Activation::Activation(std::string name, cudnnActivationMode_t mode, float coef)
{
	name_ = name;
	mode_ = mode;
	coef_ = coef;

	cudnnCreateActivationDescriptor(&act_desc_);
	cudnnSetActivationDescriptor(act_desc_, mode, CUDNN_PROPAGATE_NAN, coef);
}
Activation::~Activation()
{
	cudnnDestroyActivationDescriptor(act_desc_);
}

Blob<float> * Activation::forward(Blob<float> * input, bool is_eval)
{
	if (input_ == nullptr || batch_size_ != input->getN())
	{
		input_ = input;
		input_desc_ = input->tensor();
		batch_size_ = input->getN();

		// When using the ReLU function, we can implement the Activation Layer in an in-place manner to save memory
		// ReLU 함수를 사용할 경우, in-place 방식으로 Activation Layer를 구현하여 메모리를 절약할 수 있다.
		output_ = input_; // inplace

		// non-inplace
		// if (output_ == nullptr)
		// 	output_ = new Blob<float>(input->shape());
		// else
		// 	output_->reset(input->shape());

		output_desc_ = output_->tensor();
	}

	checkCUDNN(cudnnActivationForward(cuda_->getCudnnHandle(),
										act_desc_,
										&cuda_->one,
										input_desc_,
										input->cuda(),
										&cuda_->zero,
										output_desc_,
										output_->cuda()), __FILE__, __LINE__);

	return output_;
}

Blob<float> * Activation::backward(Blob<float> * grad_output, bool is_eval)
{
	if (grad_input_ == nullptr || batch_size_ != grad_output->getN())
	{
		grad_output_ = grad_output;

		// When using the ReLU function, we can implement the Activation Layer in an in-place manner to save memory
		// ReLU 함수를 사용할 경우, in-place 방식으로 Activation Layer를 구현하여 메모리를 절약할 수 있다.
		grad_input_ = input_; // inplace

		// non-inplace
		// if (grad_input_ == nullptr)
		// 	grad_input_ = new Blob<float>(input_->shape());
		// else
		// 	grad_input_->reset(input_->shape());
	}

	checkCUDNN(cudnnActivationBackward(cuda_->getCudnnHandle(),
										act_desc_,
										&cuda_->one,
										output_desc_, output_->cuda(),
										output_desc_, grad_output->cuda(),
										input_desc_, input_->cuda(),
										&cuda_->zero,
										input_desc_, grad_input_->cuda()), __FILE__, __LINE__);

	return grad_input_;
}

/********************************************************
* Softmax Definition									*
********************************************************/
Softmax::Softmax(std::string name)
{
	name_ = name;
}
Softmax::~Softmax()
{
	/* do nothing */
}
Blob<float> * Softmax::forward(Blob<float> * input, bool is_eval)
{
	if (input_ == nullptr || batch_size_ != input->getN())
	{
		input_ = input;
		input_desc_ = input->tensor();
		batch_size_ = input->getN();

		if (output_ == nullptr)
			output_ = new Blob<float>(input->shape());
		else
			output_->reset(input->shape());

		output_desc_ = output_->tensor();
	}

	checkCUDNN(cudnnSoftmaxForward(cuda_->getCudnnHandle(), CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
									&cuda_->one, input_desc_, input->cuda(),
									&cuda_->zero, output_desc_, output_->cuda()), __FILE__, __LINE__);

	return output_;
}

Blob<float> * Softmax::backward(Blob<float> * target, bool is_eval)
{
	// TODO : Change to cudnnSoftmaxBackward
	if (grad_input_ == nullptr || batch_size_ != target->getN())
	{
		if (grad_input_ == nullptr)
			grad_input_ = new Blob<float>(input_->shape());
		else
			grad_input_->reset(input_->shape());
	}

	// set grad_input_ as predict
	checkCUDA(cudaMemcpy(grad_input_->cuda(), output_->cuda(), output_->buf_size(), cudaMemcpyDeviceToDevice), __FILE__, __LINE__);

	// set grad_input = grad_input(predict) - target on target_f index
	checkCUBLAS(cublasSaxpy(cuda_->getCublasHandle(), target->len(),
							&cuda_->minus_one, target->cuda(), 1,
							grad_input_->cuda(), 1), __FILE__, __LINE__);

	// normalize the grad_output by the batch size
	int grad_output_size = target->getN() * target->getC() * target->getH() * target->getW();
	float scale = 1.f / static_cast<float>(target->getN());

	checkCUBLAS(cublasSscal(cuda_->getCublasHandle(), grad_output_size, &scale, grad_input_->cuda(), 1), __FILE__, __LINE__);

	return grad_input_;
}

float Softmax::get_loss(Blob<float> * target)
{
	return loss_.loss(output_, target);
}

float Softmax::get_accuracy(Blob<float> * target)
{
	return loss_.accuracy(output_, target);
}

/********************************************************
* Convolution Definition								*
********************************************************/
Conv2D::Conv2D(std::string name, int out_channels, int kernel_size, int stride, int padding, int dilation) :
	out_channels_(out_channels), kernel_size_(kernel_size), stride_(stride), padding_(padding), dilation_(dilation)
{
	name_ = name;

	// create cudnn container descriptor
	cudnnCreateFilterDescriptor(&filter_desc_);

	cudnnCreateConvolutionDescriptor(&conv_desc_);
	checkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc_, padding_, padding_, stride_, stride_, dilation_, dilation_,
												CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT), __FILE__, __LINE__);
}

Conv2D::~Conv2D()
{
	// distroy cudnn contrainer resources
	cudnnDestroyFilterDescriptor(filter_desc_);
	cudnnDestroyConvolutionDescriptor(conv_desc_);

	// terminate internal created blobs
	if (d_workspace != nullptr)
		cudaFree(d_workspace);
}

/*
	Memory usage in the Convolution Layer can vary significantly depending on the algorithm used. 
	Methods with lower memory usage tend to be non-deterministic, which is a disadvantage. 
	On the other hand, methods with higher memory usage are primarily deterministic 
	and offer the advantage of faster computation. 
	Currently implemented small-scale deep learning frameworks are not yet optimized for memory, 
	so we implement the Convolution Layer using the non-deterministic algorithm.

	Convolution Layer에서 사용되는 알고리즘에 따라 메모리 사용량이 거의 없거나 엄청 많아질 수 있다.
	메모리 사용량이 적은 방법은 non-deterministic하다는 단점이 있다.
	메모리 사용량이 많은 방법은 주로 deterministic하며 계산 속도가 빠르다는 장점이 있다.
	현재 구현된 소형 딥러닝 프레임워크는 아직 메모리 최적화가 되지 않았으므로, non-deterministic한 알고리즘을 사용하도록 구현한다.
*/
void Conv2D::set_workspace()
{
	size_t temp_size = 0;

	conv_fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM; // non-deterministic
	// conv_fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM; // deterministic but require some memory space

	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cuda_->getCudnnHandle(),
		input_desc_, filter_desc_, conv_desc_, output_desc_,
		conv_fwd_algo_, &temp_size), __FILE__, __LINE__);

	workspace_size = std::fmax(workspace_size, temp_size);

	// backward - filter
	conv_bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0; // non-deterministic
	// conv_bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1; // deterministic but require some memory

	checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cuda_->getCudnnHandle(),
		input_desc_, output_desc_, conv_desc_, filter_desc_,
		conv_bwd_filter_algo_, &temp_size), __FILE__, __LINE__);

	workspace_size = std::fmax(workspace_size, temp_size);

	// backward - data
	conv_bwd_data_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0; // non deterministic
	// conv_bwd_data_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1; // deterministic but require some memory

	checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cuda_->getCudnnHandle(),
		filter_desc_, output_desc_, conv_desc_, input_desc_,
		conv_bwd_data_algo_, &temp_size), __FILE__, __LINE__);

	workspace_size = std::fmax(workspace_size, temp_size);

	if (workspace_size > 0)
	{
		if (d_workspace != nullptr)
			cudaFree(d_workspace);

		checkCUDA(cudaMalloc((void**)&d_workspace, workspace_size), __FILE__, __LINE__);
	}
}

Blob<float> * Conv2D::forward(Blob<float> * input, bool is_eval)
{
	// weight and bias
	if (weights_ == nullptr)
	{
		checkCUDNN(cudnnSetFilter4dDescriptor(filter_desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
											out_channels_, input->getC(), kernel_size_, kernel_size_), __FILE__, __LINE__);

		weights_ = new Blob<float>(out_channels_, input->getC(), kernel_size_, kernel_size_);
		biases_ = new Blob<float>(1, out_channels_);
		bias_desc_ = biases_->tensor();

		/*
			IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!!
			IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!!
			IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!!
			Momentum Algorithm
		*/
		momentum_weights_ = new Blob<float>(weights_->shape());
		momentum_biases_ = new Blob<float>(biases_->shape());

		cudaMemset(momentum_weights_->cuda(), 0, momentum_weights_->len() * sizeof(float));
		cudaMemset(momentum_biases_->cuda(), 0, momentum_biases_->len() * sizeof(float));
	}

	// initialize input and output
	if (input_ == nullptr || batch_size_ != input->getN())
	{
		//initialize input
		input_ = input;
		input_desc_ = input->tensor();
		batch_size_ = input->getN();

		// initialize output
		checkCUDNN(cudnnGetConvolution2dForwardOutputDim(conv_desc_, input_desc_, filter_desc_,
			&output_size_[0], &output_size_[1], &output_size_[2], &output_size_[3]), __FILE__, __LINE__);

		if (output_ == nullptr)
			output_ = new Blob<float>(output_size_);
		else
			output_->reset(output_size_);

		output_desc_ = output_->tensor();

		// initialize workspace for cudnn
		set_workspace();

		// initialize weight
		if (load_pretrain_ && !freeze_)
		{
			if (load_parameter())
			{
				std::cout << "Error occured at Conv2D forward" << std::endl;
				exit(-1);
			}
		}
		else if (!freeze_)
		{
			init_weight_bias();
		}
		else
		{
			/* do nothing */
		}

	}

	checkCUDNN(cudnnConvolutionForward(cuda_->getCudnnHandle(),
		&cuda_->one, input_desc_, input->cuda(),
		filter_desc_, weights_->cuda(), conv_desc_, conv_fwd_algo_, d_workspace, workspace_size,
		&cuda_->zero, output_desc_, output_->cuda()), __FILE__, __LINE__);

	checkCUDNN(cudnnAddTensor(cuda_->getCudnnHandle(),
		&cuda_->one, bias_desc_, biases_->cuda(),
		&cuda_->one, output_desc_, output_->cuda()), __FILE__, __LINE__);

	return output_;
}

Blob<float> * Conv2D::backward(Blob<float> * grad_output, bool is_eval)
{
	// initialize grad_output back-propagation space
	if (grad_input_ == nullptr || batch_size_ != grad_output->getN())
	{
		grad_output_ = grad_output;
		grad_weights_ = new Blob<float>(weights_->shape());
		grad_biases_ = new Blob<float>(1, biases_->getC());

		if (grad_input_ == nullptr)
			grad_input_ = new Blob<float>(input_->shape());
		else
			grad_input_->reset(input_->shape());
	}

	// gradients of biases
	checkCUDNN(cudnnConvolutionBackwardBias(cuda_->getCudnnHandle(),
											&cuda_->one,
											output_desc_, grad_output->cuda(),
											&cuda_->zero,
											bias_desc_, grad_biases_->cuda()), __FILE__, __LINE__);

	// gradients of weights
	checkCUDNN(cudnnConvolutionBackwardFilter(cuda_->getCudnnHandle(),
											&cuda_->one,
											input_desc_, input_->cuda(),
											output_desc_, grad_output->cuda(),
											conv_desc_, conv_bwd_filter_algo_, d_workspace, workspace_size,
											&cuda_->zero, filter_desc_, grad_weights_->cuda()), __FILE__, __LINE__);

	if (!gradient_stop_)
	{
		checkCUDNN(cudnnConvolutionBackwardData(cuda_->getCudnnHandle(),
			&cuda_->one,
			filter_desc_, weights_->cuda(),
			output_desc_, grad_output->cuda(),
			conv_desc_, conv_bwd_data_algo_, d_workspace, workspace_size,
			&cuda_->zero,
			input_desc_, grad_input_->cuda()), __FILE__, __LINE__);
	}

	return grad_input_;
}

/********************************************************
* Pooling Definition									*
********************************************************/

Pooling::Pooling(std::string name, int kernel_size, int padding, int stride, cudnnPoolingMode_t mode) :
	kernel_size_(kernel_size), padding_(padding), stride_(stride), mode_(mode)
{
	name_ = name;

	cudnnCreatePoolingDescriptor(&pool_desc_);
	cudnnSetPooling2dDescriptor(pool_desc_, mode_, CUDNN_PROPAGATE_NAN, kernel_size_, kernel_size_,
								padding_, padding_, stride_, stride_);
}

Pooling::~Pooling()
{
	cudnnDestroyPoolingDescriptor(pool_desc_);
}

Blob<float> * Pooling::forward(Blob<float> * input, bool is_eval)
{
	if (input_ == nullptr || batch_size_ != input->getN())
	{
		input_ = input;

		// resource initialize
		input_desc_ = input_->tensor();
		batch_size_ = input_->getN();

		// setting output
		cudnnGetPooling2dForwardOutputDim(pool_desc_, input_desc_,
			&output_size_[0], &output_size_[1], &output_size_[2], &output_size_[3]);

		if (output_ == nullptr)
			output_ = new Blob<float>(output_size_);
		else
			output_->reset(output_size_);

		output_desc_ = output_->tensor();
	}

	cudnnPoolingForward(cuda_->getCudnnHandle(), pool_desc_,
		&cuda_->one, input_desc_, input_->cuda(),
		&cuda_->zero, output_desc_, output_->cuda());

	return output_;
}

Blob<float> * Pooling::backward(Blob<float> * grad_output, bool is_eval)
{
	if (grad_input_ == nullptr || batch_size_ != grad_output->getN())
	{
		grad_output_ = grad_output;

		if (grad_input_ == nullptr)
			grad_input_ = new Blob<float>(input_->shape());
		else
			grad_input_->reset(input_->shape());
	}

	checkCUDNN(cudnnPoolingBackward(cuda_->getCudnnHandle(), pool_desc_,
		&cuda_->one,
		output_desc_, output_->cuda(),
		output_desc_, grad_output->cuda(),
		input_desc_, input_->cuda(),
		&cuda_->zero,
		input_desc_, grad_input_->cuda()), __FILE__, __LINE__);

	return grad_input_;
}

/********************************************************
* Dropout Definition									*
********************************************************/
Dropout::Dropout(std::string name, float drop_rate)
{
	name_ = name;
	drop_rate_ = drop_rate;

	checkCUDNN(cudnnCreateDropoutDescriptor(&dropout_desc_), __FILE__, __LINE__);
	checkCUDNN(cudnnCreateTensorDescriptor(&dropout_in_out_desc_), __FILE__, __LINE__);
}
Dropout::~Dropout()
{
	cudnnDestroyDropoutDescriptor(dropout_desc_);
	cudnnDestroyTensorDescriptor(dropout_in_out_desc_);
}
void Dropout::set_workspace(int batch_size, int channel, int height, int width)
{
	input_size_ = output_size_ = batch_size * channel * width * height;

	checkCUDNN(cudnnSetTensor4dDescriptor(dropout_in_out_desc_,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			batch_size,
			channel,
			height,
			width
			), __FILE__, __LINE__);

	checkCUDNN(cudnnDropoutGetStatesSize(cuda_->getCudnnHandle(),
			&dropout_state_size_), __FILE__, __LINE__);

	checkCUDNN(cudnnDropoutGetReserveSpaceSize(dropout_in_out_desc_,
			&dropout_reserve_size_), __FILE__, __LINE__);

	cudaMalloc(&states_, dropout_state_size_);
	cudaMalloc(&dropout_reserve_space_, dropout_reserve_size_);

	checkCUDNN(cudnnSetDropoutDescriptor(dropout_desc_,
			cuda_->getCudnnHandle(),
			drop_rate_,
			states_,
			dropout_state_size_,
			2030), __FILE__, __LINE__);
}
Blob<float> * Dropout::forward(Blob<float> * input, bool is_eval)
{

	if(input_ == nullptr || batch_size_ != input->getN())
	{
		input_ = input;

		//resource initialize
		input_desc_ = input->tensor();
		batch_size_ = input_->getN();

		// initialize workspace for Dropout
		set_workspace(input->getN(), input->getC(), input->getH(), input->getW());

		if (output_ == nullptr)
			output_ = new Blob<float>(input_->shape());
		else
			output_->reset(input_->shape());

		output_desc_ = output_->tensor();
	}

	if(!is_eval) // if is_eval is false
	{
		checkCUDNN(cudnnDropoutForward(cuda_->getCudnnHandle(),
					dropout_desc_,
					dropout_in_out_desc_,
					input->cuda(), // different pointer between test loader and train loader!
					dropout_in_out_desc_,
					output_->cuda(),
					dropout_reserve_space_,
					dropout_reserve_size_), __FILE__, __LINE__);

		return output_;
	}
	else
	{

		return input;
	}
}

Blob<float> * Dropout::backward(Blob<float> * grad_output, bool is_eval)
{
	if(grad_input_ == nullptr || batch_size_ != grad_output->getN())
	{
		grad_output_ = grad_output;
		batch_size_ = grad_output->getN();

		if (grad_input_ == nullptr)
			grad_input_ = new Blob<float>(input_->shape());
		else
			grad_input_->reset(input_->shape());
	}

	if(!is_eval) // if is_eval is false
	{
		checkCUDNN(cudnnDropoutBackward(cuda_->getCudnnHandle(),
				dropout_desc_,
				dropout_in_out_desc_,
				grad_output->cuda(), /*grad_output_*/
				dropout_in_out_desc_,
				grad_input_->cuda(),
				dropout_reserve_space_,
				dropout_reserve_size_), __FILE__, __LINE__);

		return grad_input_;
	}
	else
	{
		return grad_output;
	}

}

/********************************************************
* BatchNormalization Definition						*
********************************************************/
BatchNormalization::BatchNormalization(std::string name, cudnnBatchNormMode_t bn_mode)
{
	name_ = name;
	batchnorm_mode_ = bn_mode;

	checkCUDNN(cudnnCreateTensorDescriptor(&batchnorm_in_out_descriptor_), __FILE__, __LINE__);
	checkCUDNN(cudnnCreateTensorDescriptor(&bnScaleBiasMeanVarDesc_), __FILE__, __LINE__);
}
BatchNormalization::~BatchNormalization()
{
	cudnnDestroyTensorDescriptor(batchnorm_in_out_descriptor_);
	cudnnDestroyTensorDescriptor(bnScaleBiasMeanVarDesc_);
}

void BatchNormalization::set_workspace(int batch_size, int channel, int height, int width)
{
	checkCUDNN(cudnnSetTensor4dDescriptor(batchnorm_in_out_descriptor_,
										CUDNN_TENSOR_NCHW,
										CUDNN_DATA_FLOAT,
										batch_size,
										channel,
										height,
										width),
										__FILE__, __LINE__);

	checkCUDNN(cudnnDeriveBNTensorDescriptor(bnScaleBiasMeanVarDesc_,
											batchnorm_in_out_descriptor_,
											batchnorm_mode_),
											__FILE__, __LINE__);

	if(batchnorm_mode_ == CUDNN_BATCHNORM_PER_ACTIVATION)
	{
		int size_D = channel * height * width;

		checkCUDA(cudaMalloc(&resultRunningMean_, size_D * sizeof(float)), __FILE__, __LINE__);
		checkCUDA(cudaMalloc(&resultRunningVariance_, size_D * sizeof(float)), __FILE__, __LINE__);
		checkCUDA(cudaMalloc(&resultSaveMean_, size_D * sizeof(float)), __FILE__, __LINE__);
		checkCUDA(cudaMalloc(&resultSaveInvVariance_, size_D * sizeof(float)), __FILE__, __LINE__);

		initKernel<<<size_D / 1024 + 1, 1024>>>(resultRunningMean_, cuda_->zero, size_D);
		initKernel<<<size_D / 1024 + 1, 1024>>>(resultRunningVariance_, cuda_->one, size_D);
	}
	else
	{
		int size_D = channel;

		checkCUDA(cudaMalloc(&resultRunningMean_, size_D * sizeof(float)), __FILE__, __LINE__);
		checkCUDA(cudaMalloc(&resultRunningVariance_, size_D * sizeof(float)), __FILE__, __LINE__);
		checkCUDA(cudaMalloc(&resultSaveMean_, size_D * sizeof(float)), __FILE__, __LINE__);
		checkCUDA(cudaMalloc(&resultSaveInvVariance_, size_D * sizeof(float)), __FILE__, __LINE__);

		initKernel<<<size_D / 1024 + 1, 1024>>>(resultRunningMean_, cuda_->zero, size_D);
		initKernel<<<size_D / 1024 + 1, 1024>>>(resultRunningVariance_, cuda_->one, size_D);
	}

}

Blob<float> * BatchNormalization::forward(Blob<float> * input, bool is_eval)
{
	// initialize weight and bias
	if (weights_ == nullptr)
	{
		// setup parameter size information
		input_size_ = output_size_ = input->getC() * input->getH() * input->getW();

		if(batchnorm_mode_ == CUDNN_BATCHNORM_PER_ACTIVATION)
		{
			int size_D = input->getC() * input->getH() * input->getW();

			// bnScale
			weights_ = new Blob<float>(1, size_D); // 1 CHW 1 1
			// bnBias
			biases_ = new Blob<float>(1, size_D); // 1 CHW 1 1

			initKernel<<<size_D / 1024 + 1, 1024>>>(weights_->cuda(), cuda_->one, size_D);
			initKernel<<<size_D / 1024 + 1, 1024>>>(biases_->cuda(), cuda_->zero, size_D);

			/*
				IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!!
				IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!!
				IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!!

				Momentum Algorithm
			*/
			momentum_weights_ = new Blob<float>(weights_->shape());
			momentum_biases_ = new Blob<float>(biases_->shape());

			cudaMemset(momentum_weights_->cuda(), 0, momentum_weights_->len() * sizeof(float));
			cudaMemset(momentum_biases_->cuda(), 0, momentum_biases_->len() * sizeof(float));
		}
		else
		{
			int size_D = input->getC();

			// bnScale
			weights_ = new Blob<float>(1, size_D); // 1 C 1 1
			// bnBias
			biases_ = new Blob<float>(1, size_D); // 1 C 1 1

			initKernel<<<size_D / 1024 + 1, 1024>>>(weights_->cuda(), cuda_->one, size_D);
			initKernel<<<size_D / 1024 + 1, 1024>>>(biases_->cuda(), cuda_->zero, size_D);

			/*
				IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!!
				IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!!
				IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!! IMPORTANT!!

				Momentum Algorithm
			*/
			momentum_weights_ = new Blob<float>(weights_->shape());
			momentum_biases_ = new Blob<float>(biases_->shape());

			cudaMemset(momentum_weights_->cuda(), 0, momentum_weights_->len() * sizeof(float));
			cudaMemset(momentum_biases_->cuda(), 0, momentum_biases_->len() * sizeof(float));
		}
	}

	// initialize input and output
	if (input_ == nullptr || batch_size_ != input->getN())
	{
		input_ = input;
		batch_size_ = input->getN();

		if (output_ == nullptr)
			output_ = new Blob<float>(input->shape());
		else
			output_->reset(input->shape());

		if (d_one_vec != nullptr)
			cudaFree(d_one_vec);
		checkCUDA(cudaMalloc((void**)&d_one_vec, sizeof(float) * batch_size_), __FILE__, __LINE__);
		initKernel<<< batch_size_ / 1024 + 1, 1024 >>>(d_one_vec, cuda_->one, batch_size_);

		// IMPORTANT
		set_workspace(input->getN(), input->getC(), input->getH(), input->getW());
	}

	/*
		The cuDNN API documentation suggests using the Cumulative Moving Average approach with:
			>> exponentialAverageFactor_ = 1.0 / nth_call_, 
		as mentioned in:
			https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnBatchNormalizationForwardTraining.
		However, this resulted in very slow training speed.
		Interestingly, the implementation code in PyTorch suggests using a constant of 0.1:
			>> exponentialAverageFactor_ = 0.1f;.
		This approach has been found to improve both training speed and stability

		cudnn API 문서에서는 Cumulative Moving Average 방식을 사용할 것을 제안한다.
			>> exponentialAverageFactor_ = 1.0 / nth_call_;
			https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnBatchNormalizationForwardTraining
		하지만, 훈련 속도가 매우 느렸다.
		오히려 Pytorch의 구현 코드에선 0.1 상수를 사용하도록 제안한다.
			>> exponentialAverageFactor_ = 0.1f;
		이것은 훈련 속도와 안정성을 높였다.
	*/
	if(!is_eval) // Train Mode
	{
		nth_call_ += 1.0;
		exponentialAverageFactor_ = 0.1f; // 1.0 / nth_call_;

		checkCUDNN(cudnnBatchNormalizationForwardTraining(cuda_->getCudnnHandle(),
														batchnorm_mode_,
														&cuda_->one,
														&cuda_->zero,
														batchnorm_in_out_descriptor_,
														input->cuda(),
														batchnorm_in_out_descriptor_,
														output_->cuda(),
														bnScaleBiasMeanVarDesc_,
														weights_->cuda(),
														biases_->cuda(),
														exponentialAverageFactor_,
														resultRunningMean_,
														resultRunningVariance_,
														epsilon_,
														nullptr, 	// resultSaveMean_			: set nullptr due to SpecTrain implementation
														nullptr),	// resultSaveInvVariance_	: set nullptr due to SpecTrain implementation
														__FILE__, __LINE__);
	}
	else // Infernece Mode
	{
		checkCUDNN(cudnnBatchNormalizationForwardInference(cuda_->getCudnnHandle(),
														batchnorm_mode_,
														&cuda_->one,
														&cuda_->zero,
														batchnorm_in_out_descriptor_,
														input->cuda(),
														batchnorm_in_out_descriptor_,
														output_->cuda(),
														bnScaleBiasMeanVarDesc_,
														weights_->cuda(),
														biases_->cuda(),
														resultRunningMean_,
														resultRunningVariance_,
														epsilon_),
														__FILE__, __LINE__);
	}

	return output_;

}

Blob<float> * BatchNormalization::backward(Blob<float> * grad_output, bool is_eval)
{
	if (grad_weights_ == nullptr)
	{
		grad_weights_ = new Blob<float>(weights_->shape());
		grad_biases_ = new Blob<float>(biases_->shape());
	}

	if (grad_input_ == nullptr || batch_size_ != grad_output->getN())
	{
		grad_output_ = grad_output;

		if (grad_input_ == nullptr)
			grad_input_ = new Blob<float>(input_->shape());
		else
			grad_input_->reset(input_->shape());
	}

	checkCUDNN(cudnnBatchNormalizationBackward(cuda_->getCudnnHandle(),
											batchnorm_mode_,
											&cuda_->one,
											&cuda_->zero,
											&cuda_->one,
											&cuda_->zero,
											batchnorm_in_out_descriptor_,
											input_->cuda(),
											batchnorm_in_out_descriptor_,
											grad_output->cuda(),
											batchnorm_in_out_descriptor_,
											grad_input_->cuda(),
											bnScaleBiasMeanVarDesc_,
											weights_->cuda(),
											grad_weights_->cuda(),
											grad_biases_->cuda(),
											epsilon_,
											nullptr,	// resultSaveMean_ : set nullptr due to SpecTrain implementation
											nullptr), 	// resultSaveInvVariance_ : set nullptr due to SpecTrain implementation
											__FILE__, __LINE__);

	return grad_input_;
}

/********************************************************
* ResidualLayer Definition								*
********************************************************/
ResidualLayer::ResidualLayer(std::string name, bool downsample, int out_channels, int kernel_size, int stride, int padding, int dilation) :
			out_channels_(out_channels), kernel_size_(kernel_size), stride_(stride), padding_(padding), dilation_(dilation)
{
	name_ = name;

	if(downsample)
	{
		// kernel_size = 1, stride = 2, padding = 1
		conv1 = new Conv2D("conv1_"+this->name_, this->out_channels_, this->kernel_size_, this->stride_ * 2, this->padding_);
		batch1 = new BatchNormalization("batch1_"+this->name_, CUDNN_BATCHNORM_SPATIAL);
		act1 = new Activation("act_relu1_"+this->name_, CUDNN_ACTIVATION_RELU);
		conv2 = new Conv2D("conv2_"+this->name_, this->out_channels_, this->kernel_size_, this->stride_, this->padding_);
		batch2 = new BatchNormalization("batch2_"+this->name_, CUDNN_BATCHNORM_SPATIAL);

		act2 = new Activation("act_relu2_"+this->name_, CUDNN_ACTIVATION_RELU);

		// 1x1 convolution downsampling to match output dimension
		// kernel_size = 1, stride = 2, padding = 0
		projection_conv = new Conv2D("projection_conv_"+this->name_, this->out_channels_, 1, this->stride_ * 2, 0);
		projection_act = nullptr;
		projection_batch = new BatchNormalization("projection_batch_"+this->name_, CUDNN_BATCHNORM_SPATIAL);
	}
	else
	{
		conv1 = new Conv2D("conv1_"+this->name_, this->out_channels_, this->kernel_size_, this->stride_, this->padding_);
		act1 = new Activation("act_relu1_"+this->name_, CUDNN_ACTIVATION_RELU);
		batch1 = new BatchNormalization("batch1_"+this->name_, CUDNN_BATCHNORM_SPATIAL);
		conv2 = new Conv2D("conv2_"+this->name_, this->out_channels_, this->kernel_size_, this->stride_, this->padding_);
		batch2 = new BatchNormalization("batch2_"+this->name_, CUDNN_BATCHNORM_SPATIAL);

		act2 = new Activation("act_relu2_"+this->name_, CUDNN_ACTIVATION_RELU);

		projection_conv = nullptr;
		projection_act = nullptr;
		projection_batch = nullptr;
	}
}
ResidualLayer::~ResidualLayer()
{
	delete conv1, act1, batch1, conv2, batch2, act2;
	if(projection_conv != nullptr)
	{
		delete projection_conv, projection_act, projection_batch;
	}
}

Blob<float> * ResidualLayer::forward(Blob<float> * input, bool is_eval)
{
	input_ = input;
	output_ = this->conv1->forward(input_, is_eval);
	output_ = this->batch1->forward(output_, is_eval);
	output_ = this->act1->forward(output_, is_eval);
	output_ = this->conv2->forward(output_, is_eval);
	output_ = this->batch2->forward(output_, is_eval);

	if(projection_conv != nullptr)
	{
		// projection
		proj_activation = this->projection_conv->forward(input_, is_eval);
		proj_activation = this->projection_batch->forward(proj_activation, is_eval);

		// output_ = output_ + proj;
		// GPU 수준 덧셈 cublasSaxPy
		const float scale = 1.0f;
		checkCUBLAS(cublasSaxpy(cuda_->getCublasHandle(),
								proj_activation->len(),
								&scale,
								proj_activation->cuda(),
								1,
								output_->cuda(),
								1), __FILE__, __LINE__);
	}
	else
	{
		// output_ = output_ + input_;
		// GPU 수준 덧셈 cublasSaxPy
		const float scale = 1.0f;
		checkCUBLAS(cublasSaxpy(cuda_->getCublasHandle(),
								input_->len(),
								&scale,
								input_->cuda(),
								1,
								output_->cuda(),
								1), __FILE__, __LINE__);
	}

	output_ = this->act2->forward(output_, is_eval);

	return output_;
}
Blob<float> * ResidualLayer::backward(Blob<float> * grad_output, bool is_eval)
{
	grad_output_ = grad_output;
	grad_output_ = this->act2->backward(grad_output_, is_eval);

	grad_input_ = this->batch2->backward(grad_output_, is_eval);
	grad_input_ = this->conv2->backward(grad_input_, is_eval);
	grad_input_ = this->act1->backward(grad_input_, is_eval);
	grad_input_ = this->batch1->backward(grad_input_, is_eval);
	grad_input_ = this->conv1->backward(grad_input_, is_eval);

	if(projection_conv != nullptr)
	{
		// projection backward
		proj_gradient = this->projection_batch->backward(grad_output_, is_eval);
		proj_gradient = this->projection_conv->backward(proj_gradient, is_eval);

		// grad_input_ = grad_input_ + proj;
		// GPU 수준 덧셈 cublasSaxPy
		const float scale = 1.0f;
		checkCUBLAS(cublasSaxpy(cuda_->getCublasHandle(),
								proj_gradient->len(),
								&scale,
								proj_gradient->cuda(),
								1,
								grad_input_->cuda(),
								1), __FILE__, __LINE__);
	}
	else
	{
		// output_ = output_ + input_;
		// GPU 수준 덧셈 cublasSaxPy
		const float scale = 1.0f;
		checkCUBLAS(cublasSaxpy(cuda_->getCublasHandle(),
								grad_output_->len(),
								&scale,
								grad_output_->cuda(),
								1,
								grad_input_->cuda(),
								1), __FILE__, __LINE__);
	}

	return grad_input_;
}
void ResidualLayer::update_weight_bias(float learning_rate)
{
	this->conv1->update_weight_bias(learning_rate);
	this->batch1->update_weight_bias(learning_rate);
	this->conv2->update_weight_bias(learning_rate);
	this->batch2->update_weight_bias(learning_rate);

	if(this->projection_conv != nullptr)
	{
		this->projection_conv->update_weight_bias(learning_rate);
		this->projection_batch->update_weight_bias(learning_rate);
	}
}

/********************************************************
* BottleNeckLayer Definition								*
********************************************************/
BottleNeckLayer::BottleNeckLayer(std::string name, bool downsample, int in_channels, int out_channels, int stride) :
			in_channels_(in_channels), out_channels_(out_channels), stride_(stride)
{
	name_ = name;

	if(downsample)
	{
		// residual 1
		if(in_channels_ == 64)
		{
			this->conv1 = new Conv2D("conv1_"+this->name_, this->in_channels_, 1, 1, 0); // 64 -> 64, kernel = 1, stride = 1, padding = 0
			this->batch1 = new BatchNormalization("batch1_"+this->name_, CUDNN_BATCHNORM_SPATIAL);
			this->act1 = new Activation("relu1_"+this->name_, CUDNN_ACTIVATION_RELU);
			
			this->conv2 = new Conv2D("conv2_"+this->name_, this->in_channels_, 3, 1, 1); // 64 -> 64, kernel = 3, stride = 1, padding = 1
			this->batch2 = new BatchNormalization("batch2_"+this->name_, CUDNN_BATCHNORM_SPATIAL);
			this->act2 = new Activation("relu2_"+this->name_, CUDNN_ACTIVATION_RELU);

			this->conv3 = new Conv2D("conv3_"+this->name_, this->out_channels_, 1, 1, 0); // 64 -> 256, kernel = 1, stride = 1, padding = 0
			this->batch3 = new BatchNormalization("batch3_"+this->name_, CUDNN_BATCHNORM_SPATIAL);
			this->act3 = new Activation("relu3_"+this->name_, CUDNN_ACTIVATION_RELU);

			projection_conv = new Conv2D("projection_conv_"+this->name_, this->out_channels_, 1, 1, 0);	// 64 -> 256, kernel = 1, stride = 1, padding = 0
			projection_batch = new BatchNormalization("projection_batch_"+this->name_, CUDNN_BATCHNORM_SPATIAL);
		}
		else // residual4, 8, 14
		{
			// residual4 : in_channels = 256, out_channels = 512
			// residual8 : in_channels = 512, out_channels = 1024
			// residual14 : in_channels = 1024, out_channels = 2048
			// 이하 주석은 residual4를 기준으로 처리.
			this->conv1 = new Conv2D("conv1_"+this->name_, this->in_channels_ / 2, 1, 1, 0); // 256 -> 128, kernel = 1, stride = 1, padding = 0
			this->batch1 = new BatchNormalization("batch1_"+this->name_, CUDNN_BATCHNORM_SPATIAL);
			this->act1 = new Activation("relu1_"+this->name_, CUDNN_ACTIVATION_RELU);
			
			this->conv2 = new Conv2D("conv2_"+this->name_, this->in_channels_ / 2, 3, 2, 1); // 128 -> 128, kernel = 3, stride = 2, padding = 1
			this->batch2 = new BatchNormalization("batch2_"+this->name_, CUDNN_BATCHNORM_SPATIAL);
			this->act2 = new Activation("relu2_"+this->name_, CUDNN_ACTIVATION_RELU);

			this->conv3 = new Conv2D("conv3_"+this->name_, this->out_channels_, 1, 1, 0); // 128 -> 512, kernel = 1, stride = 1, padding = 0
			this->batch3 = new BatchNormalization("batch3_"+this->name_, CUDNN_BATCHNORM_SPATIAL);
			this->act3 = new Activation("relu3_"+this->name_, CUDNN_ACTIVATION_RELU);

			projection_conv = new Conv2D("projection_conv_"+this->name_, this->out_channels_, 1, 2, 0);	// 256 -> 512, kernel = 1, stride = 2, padding = 0
			projection_batch = new BatchNormalization("projection_batch_"+this->name_, CUDNN_BATCHNORM_SPATIAL);
		}
	}
	else
	{
		// residual2 : in_channels = 256, out_channels = 256
		// 이하 주석은 residual2를 기준으로 처리. 다만, 다른 layer들에 대해서도 검증할 것.
		this->conv1 = new Conv2D("conv1_"+this->name_, this->in_channels_ / 4, 1, 1, 0); // 256 -> 64, kernel = 1, stride = 1, padding = 0
		this->batch1 = new BatchNormalization("batch1_"+this->name_, CUDNN_BATCHNORM_SPATIAL);
		this->act1 = new Activation("relu1_"+this->name_, CUDNN_ACTIVATION_RELU);
		
		this->conv2 = new Conv2D("conv2_"+this->name_, this->in_channels_ / 4, 3, 1, 1); // 64 -> 64, kernel = 3, stride = 1, padding = 1
		this->batch2 = new BatchNormalization("batch2_"+this->name_, CUDNN_BATCHNORM_SPATIAL);
		this->act2 = new Activation("relu2_"+this->name_, CUDNN_ACTIVATION_RELU);

		this->conv3 = new Conv2D("conv3_"+this->name_, this->out_channels_, 1, 1, 0); // 64 -> 256, kernel = 1, stride = 1, padding = 0
		this->batch3 = new BatchNormalization("batch3_"+this->name_, CUDNN_BATCHNORM_SPATIAL);
		this->act3 = new Activation("relu3_"+this->name_, CUDNN_ACTIVATION_RELU);
	}
}

BottleNeckLayer::~BottleNeckLayer()
{
	delete conv1;
	delete act1;
	delete batch1;

	delete conv2;
	delete act2;
	delete batch2;

	delete conv3;
	delete batch3;

	if(projection_conv != nullptr)
	{
		delete projection_conv;
		delete projection_batch;

		if(projection_act != nullptr)
			delete projection_act;
	}
}

Blob<float> * BottleNeckLayer::forward(Blob<float> * input, bool is_eval)
{
	input_ = input;
	output_ = this->conv1->forward(input_, is_eval);
	output_ = this->batch1->forward(output_, is_eval);
	output_ = this->act1->forward(output_, is_eval);

	output_ = this->conv2->forward(output_, is_eval);
	output_ = this->batch2->forward(output_, is_eval);
	output_ = this->act2->forward(output_, is_eval);

	output_ = this->conv3->forward(output_, is_eval);
	output_ = this->batch3->forward(output_, is_eval);

	if(projection_conv != nullptr)
	{
		// projection
		proj_activation = this->projection_conv->forward(input_, is_eval);
		proj_activation = this->projection_batch->forward(proj_activation, is_eval);

		// output_ = output_ + proj; // GPU 수준 덧셈 cublasSaxPy
		const float scale = 1.0f;
		checkCUBLAS(cublasSaxpy(cuda_->getCublasHandle(),
								proj_activation->len(),
								&scale,
								proj_activation->cuda(),
								1,
								output_->cuda(),
								1), __FILE__, __LINE__);
	}
	else
	{
		// output_ = output_ + input_; // GPU 수준 덧셈 cublasSaxPy
		const float scale = 1.0f;
		checkCUBLAS(cublasSaxpy(cuda_->getCublasHandle(),
								input_->len(),
								&scale,
								input_->cuda(),
								1,
								output_->cuda(),
								1), __FILE__, __LINE__);
	}

	output_ = this->act3->forward(output_, is_eval);

	return output_;
}

Blob<float> * BottleNeckLayer::backward(Blob<float> * grad_output, bool is_eval)
{
	grad_output_ = grad_output;
	grad_output_ = this->act3->backward(grad_output_, is_eval);

	grad_input_ = this->batch3->backward(grad_output_, is_eval);
	grad_input_ = this->conv3->backward(grad_input_, is_eval);

	grad_input_ = this->act2->backward(grad_input_, is_eval);
	grad_input_ = this->batch2->backward(grad_input_, is_eval);
	grad_input_ = this->conv2->backward(grad_input_, is_eval);

	grad_input_ = this->act1->backward(grad_input_, is_eval);
	grad_input_ = this->batch1->backward(grad_input_, is_eval);
	grad_input_ = this->conv1->backward(grad_input_, is_eval);

	if(projection_conv != nullptr)
	{
		// projection backward
		proj_gradient = this->projection_batch->backward(grad_output_, is_eval);
		proj_gradient = this->projection_conv->backward(proj_gradient, is_eval);

		// grad_input_ = grad_input_ + proj; // GPU 수준 덧셈 cublasSaxPy
		const float scale = 1.0f;
		checkCUBLAS(cublasSaxpy(cuda_->getCublasHandle(),
								proj_gradient->len(),
								&scale,
								proj_gradient->cuda(),
								1,
								grad_input_->cuda(),
								1), __FILE__, __LINE__);
	}
	else
	{
		// output_ = output_ + input_; // GPU 수준 덧셈 cublasSaxPy
		const float scale = 1.0f;
		checkCUBLAS(cublasSaxpy(cuda_->getCublasHandle(),
								grad_output_->len(),
								&scale,
								grad_output_->cuda(),
								1,
								grad_input_->cuda(),
								1), __FILE__, __LINE__);
	}

	return grad_input_;
}

void BottleNeckLayer::update_weight_bias(float learning_rate)
{
	this->conv1->update_weight_bias(learning_rate);
	this->batch1->update_weight_bias(learning_rate);
	this->conv2->update_weight_bias(learning_rate);
	this->batch2->update_weight_bias(learning_rate);
	this->conv3->update_weight_bias(learning_rate);
	this->batch3->update_weight_bias(learning_rate);

	if(this->projection_conv != nullptr)
	{
		this->projection_conv->update_weight_bias(learning_rate);
		this->projection_batch->update_weight_bias(learning_rate);
	}
}
