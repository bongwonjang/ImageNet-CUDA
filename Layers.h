/*
	Based on https://github.com/PacktPublishing/Learn-CUDA-Programming, we write DNN layer classes.
	We added (Dropout, BatchNormalization, Residual, BottleNeckLayer) and modified all layer classes.
*/
#pragma once

#include <string>

#include <cuda.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include "Loss.h"
#include "Blob.h"
#include "CudaErrorHandling.h"

class Layer
{
public:
	Layer();
	~Layer();

	virtual Blob<float> *forward(Blob<float> *input, bool is_eval = false) = 0;
	virtual Blob<float> *backward(Blob<float> *grad_input, bool is_eval = false) = 0;

	std::string get_name() { return name_; }

	virtual float get_loss(Blob<float> * target);
	virtual float get_accuracy(Blob<float> *target);

	// Overriging...
	virtual void set_cuda_context(CudaContext * context) = 0;

	void set_load_pretrain() { load_pretrain_ = true; }
	void set_gradient_stop() { gradient_stop_ = true; }

	/* Weight Freeze or Unfreeze */
	void freeze() { freeze_ = true; }
	void unfreeze() { freeze_ = false; }

protected:
	// name of layer
	std::string name_;

	// Tensor descriptor for the input/output tensors
	cudnnTensorDescriptor_t input_desc_;
	cudnnTensorDescriptor_t output_desc_;

	// Weight/Bias Descriptor
	cudnnFilterDescriptor_t filter_desc_;
	cudnnTensorDescriptor_t bias_desc_;

	/*
		y = Wx + b
	*/
	// input, output Memory
	Blob<float> * input_		= nullptr;	/* x */
	Blob<float> * output_		= nullptr;	/* y */
	Blob<float> * grad_input_	= nullptr;	/* dx */
	Blob<float> * grad_output_	= nullptr;	/* dy */

	bool freeze_				= false;	/* control parameter updates */
	Blob<float> * weights_		= nullptr;	/* W */
	Blob<float> * biases_		= nullptr;	/* b */
	Blob<float> * grad_weights_ = nullptr;	/* dW */
	Blob<float> * grad_biases_	= nullptr;	/* db */

	/*
		Momentum Algorithm
		velocity is used to compute Mementum Algorithm.
	 */
	Blob<float> * momentum_weights_ = nullptr;	/* velocity : weights */
	Blob<float> * momentum_biases_	= nullptr;	/* velocity : biases */

	int batch_size_ = 0; // mini-batch size

	// initialize weights along with the input size
	void init_weight_bias(unsigned int seed = 0);
	void update_weight_bias(float learning_rate);

	// cuda handle container
	CudaContext * cuda_ = nullptr;

	// pretrain parameters
	bool load_pretrain_ = false;
	int load_parameter();
	int save_parameter();

	// gradient stop tagging
	bool gradient_stop_ = false;

	// Network can access to private variable in Layer
	friend class Network;
};

class Dense : public Layer
{
public:
	Dense(std::string name, int out_size);
	~Dense();

	Blob<float> *forward(Blob<float> *input, bool is_eval = false);
	Blob<float> *backward(Blob<float> *grad_output, bool is_eval = false);

	void set_cuda_context(CudaContext * context) { cuda_ = context; }

private:
	int input_size_ = 0;
	int output_size_ = 0;

	float * d_one_vec = nullptr;
};

class Activation : public Layer
{
public:
	Activation(std::string name, cudnnActivationMode_t mode, float coef = 0.0f);
	~Activation();

	Blob<float> * forward(Blob<float> * input, bool is_eval = false);
	Blob<float> * backward(Blob<float> * grad_output, bool is_eval = false);

	void set_cuda_context(CudaContext * context) { cuda_ = context; }

private:
	cudnnActivationDescriptor_t act_desc_;
	cudnnActivationMode_t mode_;
	float coef_;

	friend class ResidualLayer;
	friend class BottleNeckLayer;
};

class Softmax : public Layer
{
public:
	Softmax(std::string name);
	~Softmax();

	Blob<float> * forward(Blob<float> * input, bool is_eval = false);
	Blob<float> * backward(Blob<float> * target, bool is_eval = false);

	float get_loss(Blob<float> * target);
	float get_accuracy(Blob<float> *target);

	void set_cuda_context(CudaContext * context) { cuda_ = context; }

private:
	NLLLoss loss_;

};

class Conv2D : public Layer
{
public:
	Conv2D(std::string name, int out_channels, int kernel_size, int stride = 1, int padding = 0, int dilation = 1);
	~Conv2D();

	Blob<float> * forward(Blob<float> * input, bool is_eval = false);
	Blob<float> * backward(Blob<float> * grad_output, bool is_eval = false);

	void set_cuda_context(CudaContext * context) { cuda_ = context; }

private:
	int out_channels_;
	int kernel_size_;
	int stride_;
	int padding_;
	int dilation_;

	std::array<int, 4> output_size_;

	// convolution
	cudnnConvolutionDescriptor_t conv_desc_;

	cudnnConvolutionFwdAlgo_t conv_fwd_algo_;
	cudnnConvolutionBwdDataAlgo_t conv_bwd_data_algo_;
	cudnnConvolutionBwdFilterAlgo_t conv_bwd_filter_algo_;

	size_t workspace_size = 0;

	void * d_workspace = nullptr;
	void set_workspace();

	friend class ResidualLayer;
	friend class BottleNeckLayer;

};

class Pooling : public Layer
{
public:
	Pooling(std::string name, int kernel_size, int padding, int stride, cudnnPoolingMode_t mode);
	~Pooling();

	Blob<float> * forward(Blob<float> * input, bool is_eval = false);
	Blob<float> * backward(Blob<float> * grad_output, bool is_eval = false);

	void set_cuda_context(CudaContext * context) { cuda_ = context; }

private:
	int kernel_size_;
	int padding_;
	int stride_;
	cudnnPoolingMode_t mode_;

	std::array<int, 4> output_size_;
	cudnnPoolingDescriptor_t pool_desc_;
};

class Dropout : public Layer
{
public:
	Dropout(std::string name, float drop_rate);
	~Dropout();

	Blob<float> *forward(Blob<float> *input, bool is_eval = false);
	Blob<float> *backward(Blob<float> *grad_output, bool is_eval = false);

	void set_cuda_context(CudaContext * context) { cuda_ = context; }

private:
	float drop_rate_ = 0.4;
	float * states_;
	float * dropout_reserve_space_;

	int input_size_ = 0;
	int output_size_ = 0;

	cudnnDropoutDescriptor_t dropout_desc_;
	size_t dropout_state_size_;
	size_t dropout_reserve_size_;
	cudnnTensorDescriptor_t dropout_in_out_desc_;

	void set_workspace(int batch_size, int channel, int height, int width);
};

class BatchNormalization : public Layer
{
public:
	BatchNormalization(std::string name, cudnnBatchNormMode_t bn_mode);
	~BatchNormalization();

	Blob<float> *forward(Blob<float> *input, bool is_eval = false);
	Blob<float> *backward(Blob<float> *grad_output, bool is_eval = false);

	void set_cuda_context(CudaContext * context) { cuda_ = context; }

private:
	float *resultRunningMean_;
	float *resultRunningVariance_;
	float *resultSaveMean_;
	float *resultSaveInvVariance_;

	cudnnBatchNormMode_t batchnorm_mode_;
	cudnnTensorDescriptor_t batchnorm_in_out_descriptor_;
	cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc_;

	int input_size_ = 0;
	int output_size_ = 0;

	float exponentialAverageFactor_ = 0.0;
	float nth_call_ = 0.0;
	double epsilon_ = 1e-5; // Usually 1e-5

	void set_workspace(int batch_size, int channel, int height, int width);

	float * d_one_vec = nullptr;

	friend class ResidualLayer;
	friend class BottleNeckLayer;
};

class ResidualLayer : public Layer
{
public:
	ResidualLayer(std::string name, bool downsample, int out_channels, int kernel_size, int stride = 1, int padding = 0, int dilation = 1);
	~ResidualLayer();

	Blob<float> * forward(Blob<float> * input, bool is_eval = false);
	Blob<float> * backward(Blob<float> * grad_output, bool is_eval = false);
	void update_weight_bias(float learning_rate);

	void set_cuda_context(CudaContext * context)
	{
		this->cuda_ = context;
		if(conv1 != nullptr)
			conv1->cuda_ = context;
		if(act1 != nullptr)
			act1->cuda_ = context;
		if(batch1 != nullptr)
			batch1->cuda_ = context;
		if(conv2 != nullptr)
			conv2->cuda_ = context;
		if(act2 != nullptr)
			act2->cuda_ = context;
		if(batch2 != nullptr)
			batch2->cuda_ = context;
		if(projection_conv != nullptr)
			projection_conv->cuda_ = context;
		if(projection_act != nullptr)
			projection_act->cuda_ = context;
		if(projection_batch != nullptr)
			projection_batch->cuda_ = context;
	}

	Conv2D * conv1 = nullptr;
	Activation * act1 = nullptr;
	BatchNormalization * batch1 = nullptr;
	Conv2D * conv2 = nullptr;
	Activation * act2 = nullptr;
	BatchNormalization * batch2 = nullptr;

	// For Downsample Residual Layer Projection
	Conv2D * projection_conv = nullptr;
	Activation * projection_act = nullptr;
	BatchNormalization * projection_batch = nullptr;
	Blob<float> * proj_activation = nullptr;
	Blob<float> * proj_gradient = nullptr;

private:

	int out_channels_;
	int kernel_size_;
	int stride_;
	int padding_;
	int dilation_;

	std::array<int, 4> output_size_;
};

class BottleNeckLayer : public Layer
{
public:
	BottleNeckLayer(std::string name, bool downsample, int in_channels, int out_channels, int stride = 1);
	~BottleNeckLayer();

	Blob<float> * forward(Blob<float> * input, bool is_eval = false);
	Blob<float> * backward(Blob<float> * grad_output, bool is_eval = false);
	void update_weight_bias(float learning_rate);

	void set_cuda_context(CudaContext * context)
	{
		this->cuda_ = context;
		if(conv1 != nullptr)
			conv1->cuda_ = context;
		if(act1 != nullptr)
			act1->cuda_ = context;
		if(batch1 != nullptr)
			batch1->cuda_ = context;

		if(conv2 != nullptr)
			conv2->cuda_ = context;
		if(act2 != nullptr)
			act2->cuda_ = context;
		if(batch2 != nullptr)
			batch2->cuda_ = context;

		if(conv3 != nullptr)
			conv3->cuda_ = context;
		if(act3 != nullptr)
			act3->cuda_ = context;
		if(batch3 != nullptr)
			batch3->cuda_ = context;

		if(projection_conv != nullptr)
			projection_conv->cuda_ = context;
		if(projection_act != nullptr)
			projection_act->cuda_ = context;
		if(projection_batch != nullptr)
			projection_batch->cuda_ = context;
	}

	Conv2D * conv1 = nullptr;
	Activation * act1 = nullptr;
	BatchNormalization * batch1 = nullptr;

	Conv2D * conv2 = nullptr;
	Activation * act2 = nullptr;
	BatchNormalization * batch2 = nullptr;

	Conv2D * conv3 = nullptr;
	Activation * act3 = nullptr;
	BatchNormalization * batch3 = nullptr;

	// For Downsample Residual Layer Projection
	Conv2D * projection_conv = nullptr;
	Activation * projection_act = nullptr;
	BatchNormalization * projection_batch = nullptr;
	Blob<float> * proj_activation = nullptr;
	Blob<float> * proj_gradient = nullptr;

private:
	int in_channels_;
	int out_channels_;
	int kernel_size_;
	int stride_;
	int padding_;
	int dilation_;

	std::array<int, 4> output_size_;
};
