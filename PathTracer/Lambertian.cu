#include "Lambertian.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void constructEnvironmentGPU_Lambertian(Material** this_d, Vec3 albedo)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		(*this_d) = new Lambertian(albedo);
	}
}

__global__ void destroyEnvironmentGPU_Lambertian(Material** this_d)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		delete (*this_d);
	}
}

Lambertian::Lambertian(Vec3 albedo) : albedo(albedo)
{
#ifndef __CUDA_ARCH__
	constructEnvironment();
#endif
}

Lambertian::~Lambertian()
{
#ifndef __CUDA_ARCH__
	destroyEnvironment();
#endif
}

__host__ __device__ bool Lambertian::Scatter(unsigned int* seed, Ray3& ray, const Vec3& point, const Vec3& normal, Vec3& attenuation) const
{
	Vec3 dir = (normal + Vec3::RandomUnitVector(seed));
	ray = Ray3(point, dir);
	attenuation = albedo;
	return true;
}

__host__ void Lambertian::constructEnvironment()
{
	cudaMalloc(&this_d, sizeof(Material**));
	constructEnvironmentGPU_Lambertian<<<1, 1>>>(this_d, albedo);
	cudaDeviceSynchronize();
}

__host__ void Lambertian::destroyEnvironment()
{
	destroyEnvironmentGPU_Lambertian<<<1, 1>>>(this_d);
	cudaFree(this_d);
	cudaDeviceSynchronize();
}