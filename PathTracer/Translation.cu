#include "Translation.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void constructEnvironmentGPU_Translation(Hittable** this_d, Vec3 offset, Hittable** hittable_d)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		(*this_d) = new Translation(offset, hittable_d);
	}
}

__global__ void destroyEnvironmentGPU_Translation(Hittable** this_d)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		delete (*this_d);
	}
}

Translation::Translation(Vec3 offset, Hittable** hittable_d) : offset(offset), hittable_d(hittable_d)
{
#ifndef __CUDA_ARCH__
	constructEnvironment();
#endif
}

Translation::~Translation()
{
#ifndef __CUDA_ARCH__
	destroyEnvironment();
#endif
}

bool Translation::Hit(const Ray3& ray, float tMin, float tMax, HitRecord& hRec) const
{
	Ray3 moved = Ray3(ray.Origin() - offset, ray.Direction());
	if ((*hittable_d)->Hit(moved, tMin, tMax, hRec))
	{
		hRec.SetPoint(hRec.GetPoint() + offset);
		return true;
	}
	else
	{
		return false;
	}
}

void Translation::constructEnvironment()
{
	cudaMalloc(&this_d, sizeof(Hittable**));
	constructEnvironmentGPU_Translation<<<1, 1>>>(this_d, offset, hittable_d);
	cudaDeviceSynchronize();
}

void Translation::destroyEnvironment()
{
	destroyEnvironmentGPU_Translation<<<1, 1>>>(this_d);
	cudaFree(this_d);
	cudaDeviceSynchronize();
}