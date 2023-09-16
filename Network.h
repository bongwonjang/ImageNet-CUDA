#pragma once

#include <string>
#include <vector>

#include <cudnn.h>

#include "CudaErrorHandling.h"
#include "Loss.h"
#include "Layers.h"

/*
	- Network class
		A class responsible for the deep learning model.
		It offers forward pass, backward pass, and update functions.
		
	- Network 클래스
		딥러닝 모델을 담당하는 클래스.
		순전파, 역전파, 모델 파라미터 업데이트를 전부 총괄한다.
*/
class Network
{
public:
	Network();
	~Network();

	void add_layer(Layer * layer);
	
	Blob<float> * forward(Blob<float> * input, bool is_eval = false);
	Blob<float> * backward(Blob<float> * input = nullptr, bool is_eval = false);
	void update(float learning_reate = 0.1f);

	int load_pretrain();
	int write_file();

	float loss(Blob<float> * target);
	int get_accuracy(Blob<float> *target);

	void cuda();
	void train();
	void test();

	Blob <float> * output_;

	std::vector<Layer *> layers();

private:
	std::vector<Layer *> layers_;
	CudaContext * cuda_ = nullptr;
	std::string phase_ = "training";
};
