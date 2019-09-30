#ifndef HITTABLE_H
#define HITTABLE_H

#include "Common.h"
#include "HitRecord.h"

#include <cuda_runtime.h>

class API Hittable
{
public:
	__host__ __device__ virtual ~Hittable() { }

	__host__ __device__ Hittable(const Hittable&) = delete;
	__host__ __device__ Hittable& operator=(const Hittable&) = delete;

	__host__ __device__ virtual bool Hit(const Ray3& ray, float tMin, float tMax, HitRecord& hRec) const = 0;

	__host__ Hittable** GetPtrGPU() const { return this_d; }

protected:
	Hittable** this_d{ nullptr };

	__host__ __device__ Hittable() { }
};

#endif