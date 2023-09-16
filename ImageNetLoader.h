#ifndef IMAGENETLOADER_H_
#define IMAGENETLOADER_H_

#include <iostream>
#include <fstream>
#include <array>
#include <vector>
#include <algorithm>
#include <random>
#include <string>

#include "Blob.h"
#include "DataLoader.h"

#define NUM_WORKERS 4
#define IMAGENET_CLASS 1000

typedef unsigned char BYTE;

/*
	- ImageNet class
		DataLoader for ImageNet dataset.
		You should configure 'std::string imagenet_image_dir' to point at your own ImageNet dataset location.
		
	- ImageNet 클래스
		ImageNet 데이터셋에 대한 DataLoader.
		'std::string imagenet_image_dir'를 자신의 ImageNet 데이터셋 위치에 맞도록 설정해야 함.
*/
class ImageNet : public DataLoader
{
public:
	ImageNet(std::string dataset_dir) : dataset_dir_(dataset_dir) {};
	~ImageNet();

	// load train dataset
	void train(int batch_size, int sub_dir_index, bool shuffle);
	// load test dataset
	void test(int batch_size = 1);

	// increase current step index
	int next();
	int next_data();
	int next_target();

	void reset_step()
	{
		step_ = 0;
		data_step_ = 0;
		target_step_ = 0;

		batch_index_ = 0;
	}
	int get_data_step()
	{
		return data_step_ + 1;		// num = idx + 1
	}
	int get_target_step()
	{
		return target_step_ + 1;	// num = idx + 1
	}

	// returns a pointer which has input batch data
	Blob<float> * get_data() { return data_; }
	// return a pointer which has target batch data
	Blob<float> * get_target() { return target_; }

	// Non-multi-threading
	void get_batch();
	void get_batch_data();
	void get_batch_target();

	// Multi-threading
	void thread_func(int tid = -1, int batch_offset = 0);
	void run_worker();
	void join_worker();

	void load(std::string image_file_list_path, int size_of_dataset = -1);

	void create_shared_space();
	void shuffle_datasets();

	// get private variable
	int get_num_classes() { return num_classes_; }
	int get_num_datasets() { return num_datasets_; }

private:
	std::string dataset_dir_; // txt file location
	std::string imagenet_image_dir = "/media/nogoo/T7/ILSVRC2012_ImageNet"; // real 138GB imagenet file location
	std::string train_dataset_file_ = "train.txt";
	std::string test_dataset_file_ = "val.txt";

	// container
	std::vector<std::string> data_pool_;
	std::vector<std::array<float, IMAGENET_CLASS>> target_pool_;
	Blob<float> * data_ = nullptr;		// buffer
	Blob<float> * target_ = nullptr;	// buffer

	// data loader control
	// Parmaeters for simple SGD method
	int step_ = -1;

	// Parmaeters for pipeline parallelism method
	int data_step_ = -1;
	int target_step_ = -1;

	// Basic
	int single_data_size_ = 0;
	int buffer_data_size_ = 0;
	int batch_size_ = -1;
	int channels_ = 3;
	int height_ = 224;	// default imagenet classification
	int width_ = 224;	// default imagenet classification
	int num_classes_ = 1000; // default imagenet is 1K
	int num_steps_ = 0;
	int num_datasets_ = -1;
	int batch_index_ = 0;

	bool shuffle_ = false;
	bool is_train = false;

	// Thread container
	std::vector<std::thread*> threads;

};

#endif /* IMAGENETLOADER_H_ */
