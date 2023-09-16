#include "ImageNetLoader.h"
#include "CudaErrorHandling.h"
#include <string>

ImageNet::~ImageNet()
{
	delete data_;
	delete target_;
}

void ImageNet::create_shared_space()
{
	if(data_ == nullptr)
	{
		data_ = new Blob<float>(batch_size_, channels_, height_, width_);
		data_->tensor();
	}

	if(target_ == nullptr)
		target_ = new Blob<float>(batch_size_, num_classes_);
}

/*
 * MNIST, CIFAR10, CIFAR100 와는 달리, ImageNet은 load 단계에서 file_path list를 불러옴
 */
void ImageNet::load(std::string image_file_list_path, int deprecated_size_of_dataset)
{
    //////////////////////////////////////////////////////
    // File input stream
	// ./ImageNet/train.txt
	// ./ImageNet/val.txt
	std::string file_path_ = dataset_dir_ + image_file_list_path;
	std::ifstream is;
	is.open(file_path_);

	if(!is.is_open())
	{
		printf("Failed to Open\n");
		exit(-100);
	}

	if(batch_size_ < 1)
	{
		printf("Initialize batch_size_ before load!\n");
		exit(0);
	}

    //////////////////////////////////////////////////////
    // Read File Path List
	std::string line;
	int count = 0;
	while(std::getline(is, line))
	{
		data_pool_.push_back(line);
		count++;
	}
	is.close();

	num_datasets_ = count;
	single_data_size_ = width_ * height_ * channels_;
	buffer_data_size_ = width_ * height_ * channels_ * batch_size_;

	//////////////////////////////////////////////////////
    // 근데, 크기 1000의 배열을 **128만개**나 만드는 것은.. 좀 메모리 위험
    // 미리 batch 크기만큼 공간을 담고, 실시간으로 target_pool_을
    // "0으로 리셋"->"label 표시"->"0으로 리셋"-> ...
    // 이렇게 하도록 코드를 짜자.
    // 미리 batch 크기만큼 채우기

	// 근데.. 어차피 바로 buffer를 초기화할 것인데.. 이게 필요한가?
    for(int i = 0; i < batch_size_; i++)
    {
        std::array<float, 1000> target;
        std::fill(target.begin(), target.end(), 0.0f);

        target_pool_.push_back(target);
    }

	std::cout << "Finished Loading list in " << image_file_list_path << " .." << std::endl;
}

void ImageNet::shuffle_datasets()
{
	std::mt19937 g_data(2030);
	std::shuffle(std::begin(data_pool_), std::end(data_pool_), g_data);
}

void ImageNet::train(int batch_size, int sub_dir_index, bool shuffle)
{
	is_train = true;

	if (batch_size < 1)
	{
		std::cout << "batch size should be greater than 1" << std::endl;
		exit(-1);
	}

	batch_size_ = batch_size;
	shuffle_ = shuffle;
	// 'num_datasets_' is moved to "load" function

	if(!data_pool_.empty())
		data_pool_.clear();
	if(!target_pool_.empty())
		target_pool_.clear();

	load(train_dataset_file_);

	num_steps_ = num_datasets_ / batch_size_;

	if (shuffle_)
		shuffle_datasets();

	// Load Buffer data_ and target_ (CPU, GPU)
	create_shared_space();

	step_ = 0;
	data_step_ = 0;
	target_step_ = 0;
}

void ImageNet::test(int batch_size)
{
	is_train = false;

	if (batch_size < 1)
	{
		std::cout << "batch size should be greater than 1" << std::endl;
		exit(-1);
	}

	batch_size_ = batch_size;
	shuffle_ = false;
	// 'num_datasets_' is moved to "load" function

	load(test_dataset_file_);

	num_steps_ = num_datasets_ / batch_size_;

	shuffle_datasets();

	create_shared_space();

	step_ = 0;
	data_step_ = 0;
	target_step_ = 0;
}

int extract_label(std::string file_path)
{
	/////////////////////////////////////////////////
	// File Information is as follows:
    // /train/0/n01440764_2275.JPEG
    // /val/0/ILSVRC2012_val_00037375.JPEG
    int label = 0;
    int dash_count = 0;
    std::string temp = "";

    for(int i = 0; i < file_path.length(); i++)
    {
        if(file_path[i] == '/')
            dash_count++;
        else if(dash_count == 2)
        {
            temp += file_path[i];
        }
        else if(dash_count == 3)
        {
            break;
        }
    }

    return std::stoi(temp);
}

/*
	train mode get_batch() is deprecated
*/
void ImageNet::get_batch()
{
	// This function is used only for Non-Pipelining SGD.
	if(step_ < 0)
	{
		std::cout << "you must initialize dataset first.." << std::endl;
		exit(-1);
	}

	// index clipping
	int data_idx = (step_ * batch_size_) % num_datasets_;
	// prepare data blob
	int data_size = channels_ * width_ * height_;

	// TODO: dynamic augmentation
	int target_width = this->width_;
	int target_height = this->height_;
	int resize_width = 256;
	int resize_height = 256;

	for(int batch_iter = 0; batch_iter < batch_size_; batch_iter++)
	{
		//////////////////////////////////////////////////////
		// target label 처리
		// reset target_pool_ to all "0".
		std::fill(target_pool_[batch_iter].begin(), target_pool_[batch_iter].end(), 0.0f);
		int label = extract_label(data_pool_[data_idx + batch_iter]); // label 추출 (range from 0 ~ 999)
		target_pool_[batch_iter][label] = 1.0f; // "label 표시"

		//////////////////////////////////////////////////////
		// input data 처리
		std::string image_full_path = imagenet_image_dir + data_pool_[data_idx + batch_iter];

		image im = load_image(image_full_path .c_str(), 0, 0, 3);

		if(is_train)
		{
			image resize = { 0 };
			bool is_resized = false; // to avoid double free exception

			if(im.w < resize_width || im.h < resize_height)
			{
				is_resized = true;
				resize = resize_image(im, std::max(im.w, resize_width), std::max(im.h, resize_height));
			}
			else
			{
				is_resized = false;
				resize = im;
				// printf("no resize\n");
			}

			image crop = random_crop_image(resize, target_width, target_height);
			int flip = std::rand() % 2;
			if(flip)
				flip_image(crop);

			// copy to buffer
			memcpy(&(data_->ptr()[data_size * batch_iter]), crop.data, sizeof(float) * data_size);

			free_image(im);
			if(is_resized)
				free_image(resize);
			free_image(crop);
		}
		else // test
		{
			image resize = { 0 };
			bool is_resized = true; // Always resize

			is_resized = true;
			resize = resize_image(im, std::max(im.w, resize_width), std::max(im.h, resize_height));

			image crop = center_crop_image(resize, target_width, target_height);

			// copy to buffer
			memcpy(&(data_->ptr()[data_size * batch_iter]), crop.data, sizeof(float) * data_size);

			free_image(im);
			if(is_resized)
				free_image(resize);
			free_image(crop);
		}
	}

	for (int i = 0; i < batch_size_; i++)
	{
		checkCUDA(cudaMemcpy(&(data_->cuda()[data_size * i]), &(data_->ptr()[data_size * i]), sizeof(float) * data_size, cudaMemcpyHostToDevice), __FILE__, __LINE__);
		checkCUDA(cudaMemcpy(&(target_->cuda()[num_classes_ * i]), target_pool_[i].data(), sizeof(float) * num_classes_, cudaMemcpyHostToDevice), __FILE__, __LINE__);
	}
}

// Thread 관련 함수
/* multi-threading으로 이미지를 읽고, buffer를 채울 것 */
void ImageNet::thread_func(int tid, int batch_offset)
{
	int resize_width = 256;
	int resize_height = 256;

	/* 
	 * tid가 4개라면,
	 * tid:0 -> 0 ~ 7
	 * tid:1 -> 8 ~ 15
	 * tid:2 -> 16 ~ 23
	 * tid:3 -> 24 ~ 31
	 */
	for(int batch_iter = tid * batch_offset; batch_iter < (tid + 1) * batch_offset; batch_iter++)
	{
		int label = extract_label(data_pool_[batch_index_ + batch_iter]); // label 추출 (range from 0 ~ 999)
		target_->ptr()[IMAGENET_CLASS * batch_iter + label] = 1.0f; // IMPORANT! target buffer에 직접 매핑
	}

	std::mt19937 gen(batch_index_ + tid);
	std::uniform_real_distribution<float> scale_dis(0.08, 1.0);
	std::uniform_real_distribution<float> aspect_dis(0.75, 1.33);
	for(int batch_iter = tid * batch_offset; batch_iter < (tid + 1) * batch_offset; batch_iter++)
	{
		std::string image_full_path = imagenet_image_dir + data_pool_[batch_index_ + batch_iter];

		// 아예 처음부터, load_image할 때, data point를 data_buffer의 것으로 맞추면.. 추가적인 메모리 사용이 줄어들텐데..
		image im = load_image(image_full_path.c_str(), 0, 0, 3);
		if(is_train)
		{
			image randcrop = {0};
			int new_width = -1;
			int new_height = -1;
			bool skip_fallback = false;

			// RandomResizedCrop
			for(int trial = 0; trial < 10; trial++)
			{
				float random_scale = scale_dis(gen); // 0.08 ~ 1.0
				float random_aspect = aspect_dis(gen); // 0.75 ~ 1.33 = 9 /12 ~ 12 / 9

				int area = im.w * im.h;
				float target_area = area * random_scale;

				new_width = (int)(round(sqrt(target_area * random_aspect)));
				new_height = (int)(round(sqrt(target_area / random_aspect)));

				if(((0 < new_width) && (new_width <= im.w)) && ((0 < new_height) && (new_height <= im.h)))
				{
					randcrop = random_crop_image(im, new_width, new_height);
					skip_fallback = true;
					break;
				}

			}

			// Fallback
			if(skip_fallback == false)
			{
				float im_ratio = (float)im.w / (float)im.h;
				if(im_ratio < 0.75)
				{
					// e.g.) im.w = 70, im.h = 100
					// 		 -> resize to im.w = 70 and im.h = 93 : ratio is close to 0.75
					new_width = im.w;
					new_height = (int)(round(im.w / 0.75));
				}
				else if(im_ratio > 1.33)
				{
					// e.g.) im.w = 100, im.h = 70
					//		 -> resize to im.w = 93.11 and im.h = 70 : ratio is close to 1.33
					new_height = im.h;
					new_width = (int)(round(im.h * 1.33));
				}
				else
				{
					new_width = im.w;
					new_height = im.h;
				}

				randcrop = random_crop_image(im, new_width, new_height);
			}
			
			// Resize to 224 x 224
			image resize = resize_image(randcrop, this->width_, this->height_);

			int flip = std::rand() % 2;
			if(flip)
				flip_image(resize);

			image noramlized = normalize_imagenet(resize, -1, -1, this->width_, this->height_);

			memcpy(&(data_->ptr()[single_data_size_ * batch_iter]), noramlized.data, single_data_size_ * sizeof(float));

			free_image(im);
			free_image(randcrop);
			free_image(resize);
			free_image(noramlized);
		}
		else
		{
			image resize = { 0 };
			bool is_resized = true; // Always resize

			/*
				ADJUST
			*/
			is_resized = true;
			// resize = resize_image(im, std::max(im.w, resize_width), std::max(im.h, resize_height));
			resize = resize_image(im, resize_width, resize_height);

			image crop = center_crop_image(resize, this->width_, this->height_);
			image noramlized = normalize_imagenet(resize, -1, -1, this->width_, this->height_);

			memcpy(&data_->ptr()[single_data_size_ * batch_iter], noramlized.data, single_data_size_ * sizeof(float));

			free_image(im);
			if(is_resized)
				free_image(resize);
			free_image(crop);
			free_image(noramlized);
		}
	}
}

void ImageNet::run_worker()
{
	if(this->batch_size_ < 0 || this->batch_size_ % NUM_WORKERS != 0)
	{
		printf("batch_size_ %d mod NUM_WORKERS %d should be 0\n", this->batch_size_, NUM_WORKERS);
		exit(-1);
	}

	/*
	 * 예를 들어, batch_size_ = 32, NUM_WORKERS = 4라면,
	 * 4개 쓰레드에 할당되는 offset 간격은 8이 된다.
	 * class member function을 쓰레드에서 사용하는 데, 약간 다른 문법을 사용한다.
	 * 	\=> https://stackoverflow.com/questions/10998780/stdthread-calling-method-of-class
	 */

	memset(this->data_->ptr(), 0, data_->buf_size());
	memset(this->target_->ptr(), 0, target_->buf_size());

	int offset = batch_size_ / NUM_WORKERS;
	for(int i = 0; i < NUM_WORKERS; i++)
		threads.push_back(new std::thread(&ImageNet::thread_func, this, i, offset)); // &ImageNet::thread_func, this, i, offset
}

void ImageNet::join_worker()
{
	for(int i = 0; i < NUM_WORKERS; i++)
		threads[i]->join();
	threads.clear();

	// 데이터 GPU로 전송
	checkCUDA(cudaMemcpy(data_->cuda(), data_->ptr(), data_->buf_size(), cudaMemcpyHostToDevice), __FILE__, __LINE__);
	checkCUDA(cudaMemcpy(target_->cuda(), target_->ptr(), target_->buf_size(), cudaMemcpyHostToDevice), __FILE__, __LINE__);

	// 다음 batch_size_ 이후 데이터를 불러오기 위해서, batch_index_를 증가.
	batch_index_+=batch_size_; // batch_index_ -> step_
	if(batch_index_ >= num_datasets_)
		batch_index_ = 0;
}

void ImageNet::get_batch_data()
{
//	if (data_step_ < 0)
//	{
//		std::cout << "you must initialize dataset first.." << std::endl;
//		exit(-1);
//	}
//
//	// index cliping
//	int data_idx = (data_step_ * batch_size_) % num_steps_;
//
//	// prepare data blob
//	int data_size = channels_ * width_ * height_;
//
//	for (int i = 0; i < batch_size_; i++)
//	{
//		checkCUDA(cudaMemcpy(&(data_->cuda()[data_size * i]), data_pool_[data_idx + i].data(), sizeof(float) * data_size, cudaMemcpyHostToDevice), __FILE__, __LINE__);
//	}
}

void ImageNet::get_batch_target()
{
//	if (target_step_ < 0)
//	{
//		std::cout << "you must initialize dataset first.." << std::endl;
//		exit(-1);
//	}
//
//	// index cliping
//	int data_idx = (target_step_ * batch_size_) % num_steps_;
//
//	for (int i = 0; i < batch_size_; i++)
//	{
//		checkCUDA(cudaMemcpy(&(target_->cuda()[num_classes_ * i]), target_pool_[data_idx + i].data(), sizeof(float) * num_classes_, cudaMemcpyHostToDevice), __FILE__, __LINE__);
//	}
}

int ImageNet::next_data()
{
//	data_step_++;
//
//	get_batch_data();
//
//	return data_step_;
}

int ImageNet::next_target()
{
//	target_step_++;
//
//	get_batch_target();
//
//	return target_step_;
}

int ImageNet::next()
{
	step_++;

	get_batch();

	return step_;
}
