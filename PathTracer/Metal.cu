#include "Metal.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void constructEnvironmentGPU_Metal(Material** this_d, Texture** texture_d, float fuzz)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		(*this_d) = new Metal(texture_d, fuzz);
	}
}

__global__ void destroyEnvironmentGPU_Metal(Material** this_d)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		delete (*this_d);
	}
}

Metal::Metal(Texture* texture, float fuzz) : texture(texture), texture_d(texture->GetPtrGPU()), fuzz(fuzz)
{
#ifndef __CUDA_ARCH__
	constructEnvironment();
#endif
}

__device__ Metal::Metal(Texture** texture_d, float fuzz) : texture_d(texture_d), fuzz(fuzz)
{
}

Metal::~Metal()
{
#ifndef __CUDA_ARCH__
	destroyEnvironment();
#endif
}

__host__ __device__ bool Metal::Scatter(unsigned int* seed, Ray3& ray, const HitRecord& hRec, Vec3& attenuation) const
{
	Vec3 dir = Reflect(ray.Direction(), hRec.GetNormal()) + fuzz * Vec3::RandomUnitVector(seed);
	ray = Ray3(hRec.GetPoint(), dir);
#ifdef __CUDA_ARCH__
	attenuation = (*texture_d)->Value(seed, hRec.GetU(), hRec.GetV(), hRec.GetPoint());
#else
	attenuation = texture->Value(seed, hRec.GetU(), hRec.GetV(), hRec.GetPoint());
#endif
	return Vec3::Dot(dir, hRec.GetNormal()) > 0.0f;
}

__host__ __device__ Vec3 Metal::Reflect(Vec3 v, Vec3 n) const
{
	return v - 2.0f * Vec3::Dot(v, n) * n;
}

__host__ void Metal::constructEnvironment()
{
	cudaMalloc(&this_d, sizeof(Material**));
	constructEnvironmentGPU_Metal<<<1, 1>>>(this_d, texture_d, fuzz);
	cudaDeviceSynchronize();
}

__host__ void Metal::destroyEnvironment()
{
	destroyEnvironmentGPU_Metal<<<1, 1>>>(this_d);
	cudaFree(this_d);
	cudaDeviceSynchronize();
}