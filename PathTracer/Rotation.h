#ifndef ROTATION_H
#define ROTATION_H

#include "Hittable.h"

class API Rotation : public Hittable
{
public:
	__host__ Rotation(float theta, Alignment alignment, Hittable* hittable);
	__device__ Rotation(float theta, Alignment alignment, Hittable** hittable_d);
	__host__ __device__ ~Rotation();

	__host__ __device__ Rotation(const Rotation&) = delete;
	__host__ __device__ Rotation& operator=(const Rotation&) = delete;

	__host__ __device__ bool Hit(const Ray3& ray, float tMin, float tMax, HitRecord& hRec) const override;

protected:
	float theta, cosTheta, sinTheta;
	Alignment alignment;
	Hittable* hittable{ nullptr };
	Hittable** hittable_d{ nullptr };

	__host__ void constructEnvironment();
	__host__ void destroyEnvironment();
};

#endif