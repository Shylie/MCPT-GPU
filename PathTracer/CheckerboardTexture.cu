#include "CheckerboardTexture.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void constructEnvironmentGPU_CheckerboardTexture(Texture** this_d, Texture** a_d, Texture** b_d, Vec3 offset, Vec3 frequency)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		(*this_d) = new CheckerboardTexture(a_d, b_d, offset, frequency);
	}
}

__global__ void destroyEnvironmentGPU_CheckerboardTexture(Texture** this_d)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		delete (*this_d);
	}
}

CheckerboardTexture::CheckerboardTexture(Texture* a, Texture* b, Vec3 offset, Vec3 frequency) : a(a), b(b), a_d(a->GetPtrGPU()), b_d(b->GetPtrGPU()), offset(offset), frequency(frequency)
{
#ifndef __CUDA_ARCH__
	constructEnvironment();
#endif
}

__device__ CheckerboardTexture::CheckerboardTexture(Texture** a_d, Texture** b_d, Vec3 offset, Vec3 frequency) : a_d(a_d), b_d(b_d), offset(offset), frequency(frequency)
{
}

Vec3 CheckerboardTexture::Value(unsigned int* seed, float u, float v, const Vec3& pos) const
{
	Vec3 samplePos = (pos + offset) * frequency;
	float sines = (sin(samplePos.X) * sin(samplePos.Y) * sin(samplePos.Z)) / 2.0f + 0.5f;
	if (randXORShift(seed) < sines)
	{
#ifdef __CUDA_ARCH__
		return (*a_d)->Value(seed, u, v, pos);
#else
		return a->Value(seed, u, v, pos);
#endif
	}
	else
	{
#ifdef __CUDA_ARCH__
		return (*b_d)->Value(seed, u, v, pos);
#else
		return b->Value(seed, u, v, pos);
#endif
	}
}

void CheckerboardTexture::constructEnvironment()
{
	cudaMalloc(&this_d, sizeof(Texture**));
	constructEnvironmentGPU_CheckerboardTexture<<<1, 1>>>(this_d, a_d, b_d, offset, frequency);
	cudaDeviceSynchronize();
}

void CheckerboardTexture::destroyEnvironment()
{
	destroyEnvironmentGPU_CheckerboardTexture<<<1, 1>>>(this_d);
	cudaFree(this_d);
	cudaDeviceSynchronize();
}