################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../CudaErrorHandling.cu \
../ImageNetLoader.cu \
../Layers.cu \
../Loss.cu \
../Network.cu \
../image.cu \
../kernel.cu 

OBJS += \
./CudaErrorHandling.o \
./ImageNetLoader.o \
./Layers.o \
./Loss.o \
./Network.o \
./image.o \
./kernel.o 

CU_DEPS += \
./CudaErrorHandling.d \
./ImageNetLoader.d \
./Layers.d \
./Loss.d \
./Network.d \
./image.d \
./kernel.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/bin/nvcc -O3 -gencode arch=compute_35,code=sm_35  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/bin/nvcc -O3 --compile --relocatable-device-code=false -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


