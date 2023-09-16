#include "Network.h"

#include "CudaErrorHandling.h"
#include "Layers.h"

#include <iostream>
#include <iomanip>

Network::Network()
{
	/* do nothing */
}

Network::~Network()
{
	// destroy network
	for (auto layer : layers_)
		delete layer;

	// terminate CUDA context
	if (cuda_ != nullptr)
		delete cuda_;
}

void Network::add_layer(Layer * layer)
{
	layers_.push_back(layer);
}

Blob<float> * Network::forward(Blob<float> * input, bool is_eval)
{
	output_ = input;

	for (auto layer : layers_)
		output_ = layer->forward(output_, is_eval);

	return output_;
}

Blob<float> * Network::backward(Blob<float> * target, bool is_eval)
{
	Blob<float> * gradient = target;

	if (phase_ == "inference")
		return nullptr;

	for (auto layer = layers_.rbegin(); layer != layers_.rend(); layer++)
		gradient = (*layer)->backward(gradient, is_eval);

	return gradient;
}

void Network::update(float learning_rate)
{
	if (phase_ == "inference")
		return;

	for (auto layer : layers_)
	{
		ResidualLayer * residual_ptr = dynamic_cast<ResidualLayer *>(layer);
		BottleNeckLayer * bottleneck_ptr = dynamic_cast<BottleNeckLayer *>(layer);
		if(residual_ptr != nullptr) // ResidualLayer
		{
			residual_ptr->update_weight_bias(learning_rate);
		}
		else if(bottleneck_ptr != nullptr) // BottleNeckLayer
		{
			bottleneck_ptr->update_weight_bias(learning_rate);
		}
		else // Basic Layer
		{
			// if no parameters, then pass
			if (layer->weights_ == nullptr || layer->grad_weights_ == nullptr ||
				layer->biases_ == nullptr || layer->grad_biases_ == nullptr)
				continue;

			layer->update_weight_bias(learning_rate);
		}
	}

}

int Network::write_file()
{
	std::cout << ".. store weights to the storage .." << std::endl;
	for (auto layer : layers_)
	{
		int err = layer->save_parameter();

		if (err != 0)
		{
			std::cout << "-> ERROR CODE : " << err << std::endl;
			exit(err);
		}
	}

	return 0;
}

int Network::load_pretrain()
{
	for (auto layer : layers_)
	{
		layer->set_load_pretrain();
	}

	return 0;
}

void Network::cuda()
{
	cuda_ = new CudaContext();

//	std::cout << ".. model configuration .." << std::endl;
	for (auto layer : layers_)
	{
//		std::cout << "CUDA: " << layer->get_name() << std::endl;
		layer->set_cuda_context(cuda_);
	}
}

void Network::train()
{
	phase_ = "training";

	// unfreeze all layers
	for (auto layer : layers_)
		layer->unfreeze();
}

void Network::test()
{
	phase_ = "inference";

	// freeze all layers
	for (auto layer : layers_)
		layer->freeze();
}

std::vector<Layer *> Network::layers()
{
	return layers_;
}

float Network::loss(Blob<float> * target)
{
	Layer * layer = layers_.back();
	return layer->get_loss(target);
}
int Network::get_accuracy(Blob<float> *target)
{
	Layer *layer = layers_.back();
	return layer->get_accuracy(target);
}
