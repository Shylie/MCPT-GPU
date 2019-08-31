#include "Metal.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void constructEnvironmentGPU_Metal(Material** this_d, Vec3 albedo, float fuzz)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		(*this_d) = new Metal(albedo, fuzz);
	}
}

__global__ void destroyEnvironmentGPU_Metal(Material** this_d)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		delete (*this_d);
	}
}

Metal::Metal(Vec3 albedo, float fuzz) : albedo(albedo), fuzz(fuzz)
{
#ifndef __CUDA_ARCH__
	constructEnvironment();
#endif
}

Metal::~Metal()
{
#ifndef __CUDA_ARCH__
	destroyEnvironment();
#endif
}

__host__ __device__ bool Metal::Scatter(unsigned int* seed, Ray3& ray, const Vec3& point, const Vec3& normal, Vec3& attenuation) const
{
	Vec3 dir = Reflect(ray.Direction(), normal) + fuzz * Vec3::RandomUnitVector(seed);
	ray = Ray3(point, dir);
	attenuation = albedo;
	return Vec3::Dot(dir, normal) > 0.0f;
}

__host__ __device__ Vec3 Metal::Reflect(Vec3 v, Vec3 n) const
{
	return v - 2.0f * Vec3::Dot(v, n) * n;
}

__host__ void Metal::constructEnvironment()
{
	cudaMalloc(&this_d, sizeof(Material**));
	constructEnvironmentGPU_Metal<<<1, 1>>>(this_d, albedo, fuzz);
	cudaDeviceSynchronize();
}

__host__ void Metal::destroyEnvironment()
{
	destroyEnvironmentGPU_Metal<<<1, 1>>>(this_d);
	cudaFree(this_d);
	cudaDeviceSynchronize();
}