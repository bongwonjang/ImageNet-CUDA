#pragma once

#include <thread>
#include "Blob.h"
#include "image.h"

/*
	- DataLoader class
		The top-level DataLoader class.
		We can implement a DataLoader for MNIST, CIFAR, and ImageNet datasets 
		by inheriting from this DataLoader class.
		
	- DataLoader 클래스
		최상위 DataLoader 클래스.
		DataLoader 클래스를 상속받아 MNIST, CIFAR, ImageNet 데이터셋에 대한 DataLoader를 구현할 수 있다.
*/
class DataLoader
{
public:
	DataLoader(){}
	~DataLoader(){}

	// load train dataset
	virtual void train(int batch_size = 1, int sub_dir_index = 0, bool shuffle = false) = 0;

	// load test dataset
	virtual void test(int batch_size = 1) = 0;


	virtual int next() = 0;
	virtual int next_data() = 0;
	virtual int next_target() = 0;

	virtual void reset_step() = 0;
	virtual int get_data_step() = 0;
	virtual int get_target_step() = 0;

	// returns a pointer which has input batch data
	virtual Blob<float> * get_data() = 0;

	// return a pointer which has target batch data
	virtual Blob<float> * get_target() = 0;

	/* Deprecated */
	virtual void get_batch() = 0;
	virtual void get_batch_data() = 0;
	virtual void get_batch_target() = 0;

	// Multi-threading supports
	virtual void thread_func(int tid = -1, int batch_offset = 0) = 0;
	virtual void run_worker() = 0;
	virtual void join_worker() = 0;

	// Load a dataset to CPU RAM (e.g, MNIST, CIFAR) or load a data path list (e.g, ImageNet)
	virtual void load(std::string image_file_path, int size_of_dataset = 10000) = 0;
	virtual void create_shared_space() = 0;

	// Shuffle dataset
	virtual void shuffle_datasets() = 0;

	// get private variable
	virtual int get_num_classes() = 0;
	virtual int get_num_datasets() = 0;
};
