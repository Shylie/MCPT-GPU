#ifndef SDF_HITTABLE_H
#define SDF_HITTABLE_H

#include "Hittable.h"

class API SDFHittable : public Hittable
{
public:
	__host__ __device__ SDFHittable() { }

	__host__ __device__ SDFHittable(const SDFHittable&) = delete;
	__host__ __device__ SDFHittable& operator=(const SDFHittable&) = delete;

	__host__ __device__ bool Hit(const Ray3& ray, float tMin, float tMax, HitRecord& hRec) const override;
	
	__host__ __device__ virtual float Distance(const Vec3& point) const = 0;

protected:
	__host__ __device__ virtual Material** MaterialAt(const Vec3& point) const;
	__host__ __device__ Vec3 Normal(const Vec3& point) const;
};

#endif