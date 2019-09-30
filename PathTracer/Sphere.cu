#include "Sphere.h"
#include "Material.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define PI (3.1415927f)

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

Sphere::Sphere(Vec3 center, float radius, Material* mat) : center(center), radius(radius), mat(mat), mat_d(mat != nullptr ? mat->GetPtrGPU() : nullptr)
{
#ifndef __CUDA_ARCH__
	constructEnvironment();
#endif
}

__device__ Sphere::Sphere(Vec3 center, float radius, Material** mat_d) : center(center), radius(radius), mat_d(mat_d)
{
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
			hRec.SetU(1.0f - (atan2(hRec.GetPoint().Z, hRec.GetPoint().X) + PI) / (2.0f * PI));
			hRec.SetV((asin(hRec.GetPoint().Y) + PI / 2.0f) / PI);
#ifdef __CUDA_ARCH__
			hRec.SetMaterial(mat_d);
#else
			hRec.SetMaterialHost(mat);
#endif
			return true;
		}
		temp = -b + sqrt(discriminant);
		if (temp > tMin && temp < tMax)
		{
			hRec.SetT(temp);
			hRec.SetPoint(ray.PointAt(temp));
			hRec.SetNormal(hRec.GetPoint() - center);
			hRec.SetU(1.0f - (atan2(hRec.GetPoint().Z, hRec.GetPoint().X) + PI) / (2.0f * PI));
			hRec.SetV((asin(hRec.GetPoint().Y) + PI / 2.0f) / PI);
#ifdef __CUDA_ARCH__
			hRec.SetMaterial(mat_d);
#else
			hRec.SetMaterialHost(mat);
#endif
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