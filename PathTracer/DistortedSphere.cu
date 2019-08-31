#include "DistortedSphere.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void constructEnvironmentGPU_DistortedSphere(Hittable** this_d, Vec3 center, float radius, float frequency, float amplitude, Material** mat_d)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		(*this_d) = new DistortedSphere(center, radius, frequency, amplitude, mat_d);
	}
}

__global__ void destroyEnvironmentGPU_DistortedSphere(Hittable** this_d)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		delete (*this_d);
	}
}

DistortedSphere::DistortedSphere(Vec3 center, float radius, float frequency, float amplitude, Material** mat_d) : center(center), radius(radius), frequency(frequency), amplitude(amplitude), mat_d(mat_d)
{
#ifndef __CUDA_ARCH__
	constructEnvironment();
#endif
}

DistortedSphere::~DistortedSphere()
{
#ifndef __CUDA_ARCH__
	destroyEnvironment();
#endif
}

float DistortedSphere::Distance(const Vec3& point) const
{
	return (point - center).Length() - radius + amplitude * (sin(point.X * frequency) + sin(point.Y * frequency) + sin(point.Z * frequency));
}

Material** DistortedSphere::MaterialAt(const Vec3& point) const
{
	return mat_d;
}

void DistortedSphere::constructEnvironment()
{
	cudaMalloc(&this_d, sizeof(Hittable**));
	constructEnvironmentGPU_DistortedSphere<<<1, 1>>>(this_d, center, radius, frequency, amplitude, mat_d);
	cudaDeviceSynchronize();
}

void DistortedSphere::destroyEnvironment()
{
	destroyEnvironmentGPU_DistortedSphere<<<1, 1>>>(this_d);
	cudaFree(this_d);
	cudaDeviceSynchronize();
}