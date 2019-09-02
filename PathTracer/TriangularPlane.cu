#include "TriangularPlane.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void constructEnvironmentGPU_TriangularPlane(Hittable** this_d, float a1, float b1, float a2, float b2, float a3, float b3, float k, Alignment alignment, bool autoNormal, bool invertNormal, Material** mat_d)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		(*this_d) = new TriangularPlane(a1, b1, a2, b2, a3, b3, k, alignment, autoNormal, invertNormal, mat_d);
	}
}

__global__ void destroyEnvironmentGPU_TriangularPlane(Hittable** this_d)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		delete (*this_d);
	}
}

TriangularPlane::TriangularPlane(float a1, float b1, float a2, float b2, float a3, float b3, float k, Alignment alignment, bool autoNormal, bool invertNormal, Material** mat_d) : a1(a1), b1(b1), a2(a2), b2(b2), a3(a3), b3(b3), PlaneHittable(k, alignment, autoNormal, invertNormal, mat_d)
{
#ifndef __CUDA_ARCH__
	constructEnvironment();
#endif
}

TriangularPlane::~TriangularPlane()
{
#ifndef __CUDA_ARCH__
	destroyEnvironment();
#endif
}

__host__ __device__ float sign(float ta, float tb, float p1a, float p1b, float p2a, float p2b)
{
	return (ta - p2a) * (p1b - p2b) - (p1a - p2a) * (tb - p2b);
}

bool TriangularPlane::Hit(float a, float b) const
{
	bool hasNegative = false, hasPositive = false;
	if (sign(a, b, a1, b1, a2, b2) < 0.0f)
	{
		hasNegative = true;
	}
	else
	{
		hasPositive = true;
	}

	if (sign(a, b, a2, b2, a3, b3) < 0.0f)
	{
		hasNegative = true;
	}
	else
	{
		hasPositive = true;
	}

	if (hasPositive && hasNegative) return false;

	if (sign(a, b, a3, b3, a1, b1) < 0.0f)
	{
		hasNegative = true;
	}
	else
	{
		hasPositive = true;
	}

	if (hasPositive && hasNegative) return false;

	return true;
}

void TriangularPlane::constructEnvironment()
{
	cudaMalloc(&this_d, sizeof(Hittable**));
	constructEnvironmentGPU_TriangularPlane<<<1, 1>>>(this_d, a1, b1, a2, b2, a3, b3, k, alignment, autoNormal, invertNormal, mat_d);
	cudaDeviceSynchronize();
}

void TriangularPlane::destroyEnvironment()
{
	destroyEnvironmentGPU_TriangularPlane<<<1, 1>>>(this_d);
	cudaFree(this_d);
	cudaDeviceSynchronize();
}