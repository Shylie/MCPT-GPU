#include "ConstantTexture.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void constructEnvironmentGPU_ConstantTexture(Texture** this_d, Vec3 color)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		(*this_d) = new ConstantTexture(color);
	}
}

__global__ void destroyEnvironmentGPU_ConstantTexture(Texture** this_d)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		delete (*this_d);
	}
}

ConstantTexture::ConstantTexture(Vec3 color) : color(color)
{
#ifndef __CUDA_ARCH__
	constructEnvironment();
#endif
}

ConstantTexture::~ConstantTexture()
{
#ifndef __CUDA_ARCH__
	destroyEnvironment();
#endif
}

Vec3 ConstantTexture::Value(unsigned int* seed, float u, float v, const Vec3& pos) const
{
	return color;
}

void ConstantTexture::constructEnvironment()
{
	cudaMalloc(&this_d, sizeof(Texture**));
	constructEnvironmentGPU_ConstantTexture<<<1, 1>>>(this_d, color);
	cudaDeviceSynchronize();
}

void ConstantTexture::destroyEnvironment()
{
	destroyEnvironmentGPU_ConstantTexture<<<1, 1>>>(this_d);
	cudaFree(this_d);
	cudaDeviceSynchronize();
}