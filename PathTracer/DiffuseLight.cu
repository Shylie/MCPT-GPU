#include "DiffuseLight.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void constructEnvironmentGPU_DiffuseLight(Material** this_d, Vec3 color)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		(*this_d) = new DiffuseLight(color);
	}
}

__global__ void destroyEnvironmentGPU_DiffuseLight(Material** this_d)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		delete (*this_d);
	}
}

DiffuseLight::DiffuseLight(Vec3 color) : color(color)
{
#ifndef __CUDA_ARCH__
	constructEnvironment();
#endif
}

DiffuseLight::~DiffuseLight()
{
#ifndef __CUDA_ARCH__
	destroyEnvironment();
#endif
}

__host__ __device__ Vec3 DiffuseLight::Emit() const
{
	return color;
}

__host__ void DiffuseLight::constructEnvironment()
{
	cudaMalloc(&this_d, sizeof(Material**));
	constructEnvironmentGPU_DiffuseLight<<<1, 1>>>(this_d, color);
	cudaDeviceSynchronize();
}

__host__ void DiffuseLight::destroyEnvironment()
{
	destroyEnvironmentGPU_DiffuseLight<<<1, 1>>>(this_d);
	cudaFree(this_d);
	cudaDeviceSynchronize();
}