#pragma once

#include "Blob.h"
#include "CudaErrorHandling.h"

class NLLLoss
{
public:
	NLLLoss();
	~NLLLoss();

	// Cross Entropy loss (CPU Level)
	float loss(Blob<float> * predict, Blob<float> * target_f);
	float accuracy(Blob<float> * predict, Blob<float> * target_f);

private:
	// reduced loss
	float* h_target = nullptr;
	float h_loss_ = 0.0f;
};
