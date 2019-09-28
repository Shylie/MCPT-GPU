#include "Dielectric.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void constructEnvironmentGPU_Dieletric(Material** this_d, Texture** texture_d, float refractiveIndex)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		(*this_d) = new Dieletric(texture_d, refractiveIndex);
	}
}

__global__ void destroyEnvironmentGPU_Dieletric(Material** this_d)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		delete (*this_d);
	}
}

Dieletric::Dieletric(Texture* texture, float refractiveIndex) : texture(texture), texture_d(texture->GetPtrGPU()), refractiveIndex(refractiveIndex)
{
#ifndef __CUDA_ARCH__
	constructEnvironment();
#endif
}

__device__ Dieletric::Dieletric(Texture** texture_d, float refractiveIndex) : texture_d(texture_d), refractiveIndex(refractiveIndex)
{
}

Dieletric::~Dieletric()
{
#ifndef __CUDA_ARCH__
	destroyEnvironment();
#endif
}

__host__ __device__ bool Dieletric::Scatter(unsigned int* seed, Ray3& ray, const Vec3& point, const Vec3& normal, Vec3& attenuation) const
{
#ifdef __CUDA_ARCH__
	attenuation = (*texture_d)->Value(seed, point);
#else
	attenuation = texture->Value(seed, point);
#endif
	Vec3 refracted;
	if (Vec3::Dot(ray.Direction(), normal) > 0.0f)
	{
		if (randXORShift(seed) > refractiveIndex * Schlick(Vec3::Dot(ray.Direction(), normal)))
		{
			if (Refract(ray.Direction(), -normal, refractiveIndex, refracted))
			{
				ray = Ray3(point, refracted);
				return true;
			}
		}
	}
	else
	{
		if (randXORShift(seed) > refractiveIndex * Schlick(-Vec3::Dot(ray.Direction(), normal)))
		{
			if (Refract(ray.Direction(), normal, 1.0f / refractiveIndex, refracted))
			{
				ray = Ray3(point, refracted);
				return true;
			}
		}
	}
	ray = Ray3(point, Reflect(ray.Direction(), normal));
	return true;
}

__host__ __device__ Vec3 Dieletric::Reflect(Vec3 v, Vec3 n) const
{
	return v - 2.0f * Vec3::Dot(v, n) * n;
}

__host__ __device__ bool Dieletric::Refract(Vec3 v, Vec3 n, float niOverNt, Vec3& refracted) const
{
	float dt = Vec3::Dot(v, n);
	float discriminant = 1.0f / niOverNt * niOverNt * (1.0f - dt * dt);
	if (discriminant > 0.0f)
	{
		refracted = niOverNt * (v - n * dt) - n * sqrt(discriminant);
		return true;
	}
	return false;
}

__host__ __device__ float Dieletric::Schlick(float cosine) const
{
	float r0 = (1.0f - refractiveIndex) / (1.0f + refractiveIndex);
	r0 = r0 * r0;
	return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__host__ void Dieletric::constructEnvironment()
{
	cudaMalloc(&this_d, sizeof(Material**));
	constructEnvironmentGPU_Dieletric<<<1, 1>>>(this_d, texture_d, refractiveIndex);
	cudaDeviceSynchronize();
}

__host__ void Dieletric::destroyEnvironment()
{
	destroyEnvironmentGPU_Dieletric<<<1, 1>>>(this_d);
	cudaFree(this_d);
	cudaDeviceSynchronize();
}