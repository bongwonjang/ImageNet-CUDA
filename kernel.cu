/*
	This is an ImageNet deep learning program written in C++/CUDA!

	이 소스 코드는 C++/CUDA로 작성된 ImageNet deep learning 프로그램입니다!
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include <fstream>
#include <iostream>
#include <cstdio>
#include <string>
#include <chrono>
#include <ctime>
#include <cmath>
#include <sys/time.h>

#include "Layers.h"
#include "Network.h"
#include "ImageNetLoader.h"

void print_test_loss(Network& model, DataLoader &test_data_loader);

// The most essential parameters in deep learning.
// 딥러닝에서 가장 필수적인 파라미터들
int size_of_dataset = -1;
float learning_rate = -1.0f;
int batch_size = -1;
int max_epoches = -1;

// The file name where "Test Loss" and "Test Error rate" logs are recorded
// "Test Loss"와 "Test Error rate" 로그가 저장되는 파일 이름
std::string file_path = "resultImageNet.txt";

/*
	- main function
		This function receives 4 arguments as follows.
			- argc		: 4
			- argv[0]	: ./Release/SGD
			- argv[1]	: learning rate
			- argv[2]	: batch size
			- argv[3]	: max epoches
		learning rate, batch size, and max epoches are initialize.

	- main 함수
		이 함수는 아래 설명대로 4개 인자를 받는다.
			- argc		: 4
			- argv[0]	: ./Release/SGD
			- argv[1]	: learning rate
			- argv[2]	: batch size
			- argv[3]	: max epoches	
		learning rate, batch size, 그리고 max epoches 값이 초기화된다.
*/
int main(int argc, char** argv)
{
	learning_rate = atof(argv[1]);
	batch_size = atoi(argv[2]);
	max_epoches = atoi(argv[3]);

	// current time + log file name
	// 현재 시간 + log 파일 이름
	long int now = static_cast<long int>(time(0));
	file_path = std::to_string(now) + "_" + file_path;

	cudaSetDevice(0);

	// Generate ImageNet dataloader
	// ImageNet dataloader 생성
	ImageNet train_data_loader = ImageNet("./ImageNet/");
	ImageNet test_data_loader = ImageNet("./ImageNet/");
	train_data_loader.train(batch_size, 0, true);
	test_data_loader.test(batch_size);

	// Get the training dataset size from 'train_data_loader'
	// 'train_data_loader'에서 훈련 데이터셋 크기를 알아내기
	size_of_dataset = train_data_loader.get_num_datasets();
	if(size_of_dataset <= 0)
	{
		printf("Unfinished loading datasets!\n");
		exit(-1);
	}

	// Deep learning model initialization
	// 딥러닝 모델 초기화
	Network model;

	/* ResNet-34 */
	model.add_layer(new Conv2D("conv1", 64, 7, 2, 3));
	model.add_layer(new BatchNormalization("bn", CUDNN_BATCHNORM_SPATIAL));
	model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
	model.add_layer(new Pooling("pool_max", 3, 1, 2, CUDNN_POOLING_MAX));

	/* 3 */
	model.add_layer(new ResidualLayer("residual", false, 64, 3, 1, 1));
	model.add_layer(new ResidualLayer("residual", false, 64, 3, 1, 1));
	model.add_layer(new ResidualLayer("residual", false, 64, 3, 1, 1));

	/* 4 */
	model.add_layer(new ResidualLayer("residual", true, 128, 3, 1, 1));
	model.add_layer(new ResidualLayer("residual", false, 128, 3, 1, 1));
	model.add_layer(new ResidualLayer("residual", false, 128, 3, 1, 1));
	model.add_layer(new ResidualLayer("residual", false, 128, 3, 1, 1));

	/* 6 */
	model.add_layer(new ResidualLayer("residual", true, 256, 3, 1, 1));
	model.add_layer(new ResidualLayer("residual", false, 256, 3, 1, 1));
	model.add_layer(new ResidualLayer("residual", false, 256, 3, 1, 1));
	model.add_layer(new ResidualLayer("residual", false, 256, 3, 1, 1));
	model.add_layer(new ResidualLayer("residual", false, 256, 3, 1, 1));
	model.add_layer(new ResidualLayer("residual", false, 256, 3, 1, 1));

	/* 3 */
	model.add_layer(new ResidualLayer("residual", true, 512, 3, 1, 1));
	model.add_layer(new ResidualLayer("residual", false, 512, 3, 1, 1));
	model.add_layer(new ResidualLayer("residual", false, 512, 3, 1, 1));

	model.add_layer(new Pooling("pool_adaptive", 7, 0, 7, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING));

	model.add_layer(new Dense("dense", train_data_loader.get_num_classes()));
	model.add_layer(new Softmax("softmax"));
	
	// /* ResNet-50 */
	// model.add_layer(new Conv2D("conv1", 64, 7, 2, 3));
	// model.add_layer(new BatchNormalization("bn", CUDNN_BATCHNORM_SPATIAL));
	// model.add_layer(new Activation("relu", CUDNN_ACTIVATION_RELU));
	// model.add_layer(new Pooling("pool_max", 3, 1, 2, CUDNN_POOLING_MAX));

	// /* 3 */
	// model.add_layer(new BottleNeckLayer("residual1", true, 64, 256, 1));
	// model.add_layer(new BottleNeckLayer("residual2", false, 256, 256, 1));
	// model.add_layer(new BottleNeckLayer("residual3", false, 256, 256, 1));

	// /* 4 */
	// model.add_layer(new BottleNeckLayer("residual4", true, 256, 512, 1));
	// model.add_layer(new BottleNeckLayer("residual5", false, 512, 512, 1));
	// model.add_layer(new BottleNeckLayer("residual6", false, 512, 512, 1));
	// model.add_layer(new BottleNeckLayer("residual7", false, 512, 512, 1));

	// /* 6 */
	// model.add_layer(new BottleNeckLayer("residual8", true, 512, 1024, 1));
	// model.add_layer(new BottleNeckLayer("residual9", false, 1024, 1024, 1));
	// model.add_layer(new BottleNeckLayer("residual10", false, 1024, 1024, 1));
	// model.add_layer(new BottleNeckLayer("residual11", false, 1024, 1024, 1));
	// model.add_layer(new BottleNeckLayer("residual12", false, 1024, 1024, 1));
	// model.add_layer(new BottleNeckLayer("residual13", false, 1024, 1024, 1));

	// /* 3 */
	// model.add_layer(new BottleNeckLayer("residual14", true, 1024, 2048, 1));
	// model.add_layer(new BottleNeckLayer("residual15", false, 2048, 2048, 1));
	// model.add_layer(new BottleNeckLayer("residual16", false, 2048, 2048, 1));

	// model.add_layer(new Pooling("pool_adaptive", 7, 0, 7, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING));

	// model.add_layer(new Dense("dense", train_data_loader.get_num_classes()));
	// model.add_layer(new Softmax("softmax"));

	// Assign 'cublas' and 'cudnn' handlers to each layer for training a deep learning model
	// 딥러닝 모델 훈련에 필요한 'cublas'와 'cudnn' handler를 각 layer에게 부여
	model.cuda();

	// Start train!
	// 훈련 시작!
	std::cout << "[TRAIN]" << std::endl;
	float total_elapsed_cpu_time = 0.0f;
	struct timeval begin, end;

	// Pass the memory addresses of the input data and target labels to 'train_data' 
	// and 'train_target', respectively.
	// 'train_data'와 'train_target'에 각각 입력 데이터와 타깃 라벨의 메모리 주소 전달.
	Blob<float> *train_data = train_data_loader.get_data();
	Blob<float> *train_target = train_data_loader.get_target();

	for(int epoch = 0; epoch < max_epoches; epoch++)
	{
		train_data_loader.run_worker();

		float p_learning_rate = learning_rate;

		// Step Learning rate decay. 
		// Reduce the learning rate by a factor of 10 every 30 epochs. 
		// 매 30 epoch 마다 learning rate를 10분의 1로 줄이기
		if(epoch < 30)
			p_learning_rate = learning_rate;
		else if(epoch < 60)
			p_learning_rate = learning_rate * 0.1f;
		else if(epoch < 90)
			p_learning_rate = learning_rate * 0.01f;
		else
			p_learning_rate = learning_rate * 0.001f;
		
		int total_iter = (int)(size_of_dataset / batch_size) - 1; // prevent overflow of dataloader

		gettimeofday(&begin, NULL);

		float total_loss = 0.0f;
		float tp_count = 0.0f;
		float total_count = 0.0f;

		for(int iter = 0; iter < total_iter; iter++)
		{
			// Dataloading technique inspired by Darknet. Utilizing multi-threading 
			// to simultaneously prepare the next training data while training the model.
			// Darknet에서 가져온 데이터 로딩 테크닉. 멀티 쓰레딩을 사용하여, 모델을 학습시키면서 동시에
			// 다음 훈련 데이터를 준비.
			train_data_loader.join_worker();
			train_data_loader.run_worker();

			// forward pass
			// 'false' means this model is 'train' mode, not 'eval' mode
			// 순전파
			// 'false'는 현재 모델이 'eval' 모드가 아니라 'train' 모드임을 나타냄
			model.forward(train_data, false);

			total_loss += model.loss(train_target);
			tp_count += model.get_accuracy(train_target);
			total_count += batch_size;

			/// backward pass
			// 'false' means this model is 'train' mode, not 'eval' mode
			// 역전파
			// 'false'는 현재 모델이 'eval' 모드가 아니라 'train' 모드임을 나타냄
			model.backward(train_target, false);

			// update parameter
			// 모델 파라미터 업데이트
			model.update(p_learning_rate);


			gettimeofday(&end, NULL);
			float current_elapsed_time = ((end.tv_sec - begin.tv_sec) + ((end.tv_usec - begin.tv_usec) / 1000000.0f));
			if(iter % 1000 == 0)
				printf("\ttrain iter %d in total iter %d [%.2f secs] avg_loss = %.3f accuracy = %.3f percent\n", iter, total_iter, current_elapsed_time,
						total_loss / total_count, 100.0f * tp_count / total_count);
		}

		cudaDeviceSynchronize();

		gettimeofday(&end, NULL);

		total_elapsed_cpu_time += ((end.tv_sec - begin.tv_sec) + ((end.tv_usec - begin.tv_usec) / 1000000.0f));

		// Print log
		// 로그 출력
		std::cout << "Epoch : " << epoch << "                    Training Time Elapsed : " << total_elapsed_cpu_time << std::endl;
		std::cout << "Learning rate : " << p_learning_rate << std::endl;

		std::ofstream writeFile(file_path.data(), std::ios::out | std::ios_base::app);
		if(writeFile.is_open())
		{
			writeFile << "Epoch : " << epoch << "                    Training Time Elapsed : " << total_elapsed_cpu_time << std::endl;
			writeFile << "Learning rate : " << p_learning_rate << std::endl;
			writeFile.close();
		}

		print_test_loss(model, test_data_loader);

		train_data_loader.shuffle_datasets();
		train_data_loader.reset_step();
	}

	return 0;
}

/*
	- print_test_loss function
		This function receives 2 arguments as follows.
			- model : a deep learning model
			- test_data_loader : test dataset loader

	- print_test_loss 함수
		이 함수는 아래 설명대로 2개 인자를 받는다.
			- model : 딥러닝 모델
			- test_data_loader : 테스트 데이터셋 데이터 로더
*/
void print_test_loss(Network &model, DataLoader &test_data_loader)
{
	test_data_loader.reset_step();

	int size_of_test = test_data_loader.get_num_datasets();

	float total_loss = 0;
	float tp_count = 0;

	Blob<float> *test_data = test_data_loader.get_data();
	Blob<float> *test_target = test_data_loader.get_target();
	test_data_loader.run_worker();

	int total_iter = (int)(size_of_test / batch_size) - 1; // prevent overflow of dataloader

	for(int iter = 0; iter < total_iter; iter++)
	{
		test_data_loader.join_worker();
		test_data_loader.run_worker();

		// forward pass
		// 순전파
		model.forward(test_data, true);
		total_loss += model.loss(test_target);
		tp_count += model.get_accuracy(test_target);
	}

	cudaDeviceSynchronize();

	// Calculate 'Test Loss' and 'Test Error rate (or Accuracy rate)'
	// 'Test Loss'와 'Test Error rate (또는 Accuracy rate)'를 계산
	float avg_loss = total_loss / ((total_iter + 1) * batch_size);
	float accuracy = 100.0f * tp_count / ((total_iter + 1) * batch_size);

	std::cout << "Test Loss             : " << avg_loss << std::endl;
	std::cout << "Test Accuracy Rate    : " << accuracy << " percent" << std::endl;

	std::ofstream writeFile(file_path.data(), std::ios::out | std::ios_base::app);
	if(writeFile.is_open())
	{
		writeFile << "Test Loss             : " << avg_loss << std::endl;
		writeFile << "Test Accuracy Rate    : " << accuracy << " percent" << std::endl;
		writeFile.close();
	}
}
