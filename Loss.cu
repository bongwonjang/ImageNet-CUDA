#include <cuda.h>
#include <cassert>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "Loss.h"

NLLLoss::NLLLoss() { /* nothing */ }

NLLLoss::~NLLLoss() { /* nothing */ }

float NLLLoss::loss(Blob<float> * predict, Blob<float> * target)
{
	int batch_size = target->getN();	// batch size
	int output_size = target->size();

	assert(batch_size == predict->getN());
	assert(output_size == predict->size());

	// get predicts and targets
	float* h_predict = predict->to("host");

	if(this->h_target == nullptr)
		this->h_target = new float[target->len()];
	cudaMemcpy(this->h_target, target->cuda(), target->buf_size(), cudaMemcpyDeviceToHost);

	// reset h_loss_
	h_loss_ = 0;

	// idx_output = idx_target = 0;
	for (int b = 0; b < batch_size; b++)
	{
		for (int i = 0; i < output_size; i++)
		{
			if (h_target[b * output_size + i] == 1.0f)
			{
				// 1e-7 is epsilon for no zero
				// WARNING! WARNING! WARNING! WARNING! WARNING! WARNING!
				h_loss_ +=-logf(h_predict[b * output_size + i] + 1e-7);
				break;
			}

		}
	}

	return h_loss_;
}

float NLLLoss::accuracy(Blob<float> * predict, Blob<float> * target)
{
	int batch_size = target->getN();
	int output_size = target->size();

	assert(batch_size == predict->getN());
	assert(output_size == predict->size());

	int idx_output, idx_target;
	int hit_count = 0;

	// get predicts and targets
	float *h_output = predict->to("host");

	if(this->h_target == nullptr)
		this->h_target = new float[target->len()];
	cudaMemcpy(this->h_target, target->cuda(), target->buf_size(), cudaMemcpyDeviceToHost);

	// idx_output = idx_target = 0;
	for (int b = 0; b < batch_size; b++)
	{
		idx_output = 0;
		idx_target = 0;

		for (int i = 1; i < output_size /* 10 */; i++) // MNIST
		{
			if (h_output[b * output_size + i] > h_output[b * output_size + idx_output])
				idx_output = i;

			if (h_target[b * output_size + i] > h_target[b * output_size + idx_target])
				idx_target = i;
		}

		if (idx_output == idx_target)
			hit_count++;
	}

	return hit_count;
}

