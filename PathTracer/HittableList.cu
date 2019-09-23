#include "HittableList.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void constructEnvironmentGPU_HittableList(Hittable** this_d, int numHittables, Hittable*** hittables_d)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		(*this_d) = new HittableList(numHittables, hittables_d);
	}
}

__global__ void destroyEnvironmentGPU_HittableList(Hittable** this_d)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		delete (*this_d);
	}
}

HittableList::HittableList(int numHittables, Hittable** hittables) : numHittables(numHittables), hittables(hittables), hittables_d(new Hittable**[numHittables])
{
#ifndef __CUDA_ARCH__
	for (int i = 0; i < numHittables; i++)
	{
		hittables_d[i] = hittables[i]->GetPtrGPU();
	}

	constructEnvironment();
#endif
}

__device__ HittableList::HittableList(int numHittables, Hittable*** hittables_d) : numHittables(numHittables), hittables_d(hittables_d)
{
}

HittableList::~HittableList()
{
#ifndef __CUDA_ARCH__
	destroyEnvironment();
#endif
}

bool HittableList::Hit(const Ray3& ray, float tMin, float tMax, HitRecord& hRec) const
{
	bool hit = false;
	HitRecord temp;
	for (int i = 0; i < numHittables; i++)
	{
		if ((*(hittables_d[i]))->Hit(ray, tMin, tMax, temp))
		{
			tMax = temp.GetT();
			hRec = temp;
			hit = true;
		}
	}
	return hit;
}

void HittableList::constructEnvironment()
{
	cudaMalloc(&this_d, sizeof(Hittable**));
	Hittable*** temp;
	cudaMalloc(&temp, numHittables * sizeof(Hittable**));
	device_hittables_d = temp;
	cudaMemcpy(temp, hittables_d, numHittables * sizeof(Hittable**), cudaMemcpyKind::cudaMemcpyHostToDevice);
	constructEnvironmentGPU_HittableList<<<1, 1>>>(this_d, numHittables, temp);
	cudaDeviceSynchronize();
}

void HittableList::destroyEnvironment()
{
	cudaFree(device_hittables_d);
	destroyEnvironmentGPU_HittableList<<<1, 1>>>(this_d);
	cudaFree(this_d);
	cudaDeviceSynchronize();
}