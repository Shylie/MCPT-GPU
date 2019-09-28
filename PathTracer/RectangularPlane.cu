#include "RectangularPlane.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void constructEnvironmentGPU_RectangularPlane(Hittable** this_d, float a1, float a2, float b1, float b2, float k, Alignment alignment, bool autoNormal, bool invertNormal, Material** mat_d)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		(*this_d) = new RectangularPlane(a1, a2, b1, b2, k, alignment, autoNormal, invertNormal, mat_d);
	}
}

__global__ void destroyEnvironmentGPU_RectangularPlane(Hittable** this_d)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		delete (*this_d);
	}
}

RectangularPlane::RectangularPlane(float a1, float a2, float b1, float b2, float k, Alignment alignment, bool autoNormal, bool invertNormal, Material* mat) : a1(a1), a2(a2), b1(b1), b2(b2), PlaneHittable(k, alignment, autoNormal, invertNormal, mat)
{
#ifndef __CUDA_ARCH__
	constructEnvironment();
#endif
}

__device__ RectangularPlane::RectangularPlane(float a1, float a2, float b1, float b2, float k, Alignment alignment, bool autoNormal, bool invertNormal, Material** mat_d) : a1(a1), a2(a2), b1(b1), b2(b2), PlaneHittable(k, alignment, autoNormal, invertNormal, mat_d)
{
}

RectangularPlane::~RectangularPlane()
{
#ifndef __CUDA_ARCH__
	destroyEnvironment();
#endif
}

bool RectangularPlane::Hit(float a, float b) const
{
	return a > a1 && a < a2 && b > b1 && b < b2;
}

void RectangularPlane::constructEnvironment()
{
	cudaMalloc(&this_d, sizeof(Hittable**));
	constructEnvironmentGPU_RectangularPlane<<<1, 1>>>(this_d, a1, a2, b1, b2, k, alignment, autoNormal, invertNormal, mat_d);
	cudaDeviceSynchronize();
}

void RectangularPlane::destroyEnvironment()
{
	destroyEnvironmentGPU_RectangularPlane<<<1, 1>>>(this_d);
	cudaFree(this_d);
	cudaDeviceSynchronize();
}