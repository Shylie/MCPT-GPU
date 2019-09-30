#include "DiffuseLight.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void constructEnvironmentGPU_DiffuseLight(Material** this_d, Texture** texture_d)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		(*this_d) = new DiffuseLight(texture_d);
	}
}

__global__ void destroyEnvironmentGPU_DiffuseLight(Material** this_d)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		delete (*this_d);
	}
}

DiffuseLight::DiffuseLight(Texture* texture) : texture(texture), texture_d(texture->GetPtrGPU())
{
#ifndef __CUDA_ARCH__
	constructEnvironment();
#endif
}

__device__ DiffuseLight::DiffuseLight(Texture** texture_d) : texture_d(texture_d)
{
}

DiffuseLight::~DiffuseLight()
{
#ifndef __CUDA_ARCH__
	destroyEnvironment();
#endif
}

__host__ __device__ Vec3 DiffuseLight::Emit(unsigned int* seed, const HitRecord& hRec) const
{
#ifdef __CUDA_ARCH__
	return (*texture_d)->Value(seed, hRec.GetU(), hRec.GetV(), hRec.GetPoint());
#else
	return texture->Value(seed, hRec.GetU(), hRec.GetV(), hRec.GetPoint());
#endif
}

__host__ void DiffuseLight::constructEnvironment()
{
	cudaMalloc(&this_d, sizeof(Material**));
	constructEnvironmentGPU_DiffuseLight<<<1, 1>>>(this_d, texture_d);
	cudaDeviceSynchronize();
}

__host__ void DiffuseLight::destroyEnvironment()
{
	destroyEnvironmentGPU_DiffuseLight<<<1, 1>>>(this_d);
	cudaFree(this_d);
	cudaDeviceSynchronize();
}