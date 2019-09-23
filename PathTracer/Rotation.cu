#include "Rotation.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void constructEnvironmentGPU_Rotation(Hittable** this_d, float theta, Alignment alignment, Hittable** hittable_d)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		(*this_d) = new Rotation(theta, alignment, hittable_d);
	}
}

__global__ void destroyEnvironmentGPU_Rotation(Hittable** this_d)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		delete (*this_d);
	}
}

Rotation::Rotation(float theta, Alignment alignment, Hittable* hittable) : theta(theta), cosTheta(cos(theta)), sinTheta(sin(theta)), alignment(alignment), hittable(hittable), hittable_d(hittable->GetPtrGPU())
{
#ifndef __CUDA_ARCH__
	constructEnvironment();
#endif
}

__device__ Rotation::Rotation(float theta, Alignment alignment, Hittable** hittable_d) : theta(theta), cosTheta(cos(theta)), sinTheta(sin(theta)), alignment(alignment), hittable_d(hittable_d)
{
}

Rotation::~Rotation()
{
#ifndef __CUDA_ARCH__
	destroyEnvironment();
#endif
}

bool Rotation::Hit(const Ray3& ray, float tMin, float tMax, HitRecord& hRec) const
{
	Vec3 origin = ray.Origin(), direction = ray.Direction();
	switch (alignment)
	{
	case X:
		origin.Y = cosTheta * ray.Origin().Y - sinTheta * ray.Origin().Z;
		origin.Z = sinTheta * ray.Origin().Y + cosTheta * ray.Origin().Z;
		direction.Y = cosTheta * ray.Direction().Y - sinTheta * ray.Direction().Z;
		direction.Z = sinTheta * ray.Direction().Y + cosTheta * ray.Direction().Z;
		break;
	case Y:
		origin.X = cosTheta * ray.Origin().X - sinTheta * ray.Origin().Z;
		origin.Z = sinTheta * ray.Origin().X + cosTheta * ray.Origin().Z;
		direction.X = cosTheta * ray.Direction().X - sinTheta * ray.Direction().Z;
		direction.Z = sinTheta * ray.Direction().X + cosTheta * ray.Direction().Z;
		break;
	case Z:
		origin.X = cosTheta * ray.Origin().X - sinTheta * ray.Origin().Y;
		origin.Y = sinTheta * ray.Origin().X + cosTheta * ray.Origin().Y;
		direction.X = cosTheta * ray.Direction().X - sinTheta * ray.Direction().Y;
		direction.Y = sinTheta * ray.Direction().X + cosTheta * ray.Direction().Y;
		break;
	default:
		return false;
	}
	Ray3 rotatedRay = Ray3(origin, direction);
#ifdef __CUDA_ARCH__
	if ((*hittable_d)->Hit(rotatedRay, tMin, tMax, hRec))
#else
	if (hittable->Hit(rotatedRay, tMin, tMax, hRec))
#endif
	{
		Vec3 p = hRec.GetPoint();
		Vec3 n = hRec.GetNormal();
		switch (alignment)
		{
		case X:
			p.Y = cosTheta * hRec.GetPoint().Y + sinTheta * hRec.GetPoint().Z;
			p.Z = -sinTheta * hRec.GetPoint().Y + cosTheta * hRec.GetPoint().Z;
			n.Y = cosTheta * hRec.GetNormal().Y + sinTheta * hRec.GetNormal().Z;
			n.Z = -sinTheta * hRec.GetNormal().Y + cosTheta * hRec.GetNormal().Z;
			break;
		case Y:
			p.X = cosTheta * hRec.GetPoint().X + sinTheta * hRec.GetPoint().Z;
			p.Z = -sinTheta * hRec.GetPoint().X + cosTheta * hRec.GetPoint().Z;
			n.X = cosTheta * hRec.GetNormal().X + sinTheta * hRec.GetNormal().Z;
			n.Z = -sinTheta * hRec.GetNormal().X + cosTheta * hRec.GetNormal().Z;
			break;
		case Z:
			p.X = cosTheta * hRec.GetPoint().X + sinTheta * hRec.GetPoint().Y;
			p.Y = -sinTheta * hRec.GetPoint().X + cosTheta * hRec.GetPoint().Y;
			n.X = cosTheta * hRec.GetNormal().X + sinTheta * hRec.GetNormal().Y;
			n.Y = -sinTheta * hRec.GetNormal().X + cosTheta * hRec.GetNormal().Y;
			break;
		}
		hRec.SetPoint(p);
		hRec.SetNormal(n);
		return true;
	}
	else
	{
		return false;
	}
}

void Rotation::constructEnvironment()
{
	cudaMalloc(&this_d, sizeof(Hittable**));
	constructEnvironmentGPU_Rotation<<<1, 1>>>(this_d, theta, alignment, hittable_d);
	cudaDeviceSynchronize();
}

void Rotation::destroyEnvironment()
{
	destroyEnvironmentGPU_Rotation<<<1, 1>>>(this_d);
	cudaFree(this_d);
	cudaDeviceSynchronize();
}