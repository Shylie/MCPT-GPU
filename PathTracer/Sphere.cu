#include "Sphere.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void constructEnvironmentGPU_Sphere(Hittable** this_d, Vec3 center, float radius, Material** mat_d)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		(*this_d) = new Sphere(center, radius, mat_d);
	}
}

__global__ void destroyEnvironmentGPU_Sphere(Hittable** this_d)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		delete (*this_d);
	}
}

Sphere::Sphere(Vec3 center, float radius, Material** mat_d) : center(center), radius(radius), mat_d(mat_d)
{
#ifndef __CUDA_ARCH__
	constructEnvironment();
#endif
}

Sphere::~Sphere()
{
#ifndef __CUDA_ARCH__
	destroyEnvironment();
#endif
}

bool Sphere::Hit(const Ray3& ray, float tMin, float tMax, HitRecord& hRec) const
{
	Vec3 oc = ray.Origin() - center;
	float b = Vec3::Dot(oc, ray.Direction());
	float c = oc.LengthSquared() - radius * radius;
	float discriminant = b * b - c;
	if (discriminant > 0.0f)
	{
		float temp = -b - sqrt(discriminant);
		if (temp > tMin && temp < tMax)
		{
			hRec.SetT(temp);
			hRec.SetPoint(ray.PointAt(temp));
			hRec.SetNormal(hRec.GetPoint() - center);
			hRec.SetMaterial(mat_d);
			return true;
		}
		temp = -b + sqrt(discriminant);
		if (temp > tMin && temp < tMax)
		{
			hRec.SetT(temp);
			hRec.SetPoint(ray.PointAt(temp));
			hRec.SetNormal(hRec.GetPoint() - center);
			hRec.SetMaterial(mat_d);
			return true;
		}
	}
	return false;
}

void Sphere::constructEnvironment()
{
	cudaMalloc(&this_d, sizeof(Hittable**));
	constructEnvironmentGPU_Sphere<<<1, 1>>>(this_d, center, radius, mat_d);
	cudaDeviceSynchronize();
}

void Sphere::destroyEnvironment()
{
	destroyEnvironmentGPU_Sphere<<<1, 1>>>(this_d);
	cudaFree(this_d);
	cudaDeviceSynchronize();
}