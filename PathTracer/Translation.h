#ifndef TRANSLATION_H
#define TRANSLATION_H

#include "Hittable.h"

class API Translation : public Hittable
{
public:
	__host__ Translation(Vec3 offset, Hittable* hittable);
	__device__ Translation(Vec3 offset, Hittable** hittable_d);
	__host__ __device__ ~Translation();

	__host__ __device__ Translation(const Translation&) = delete;
	__host__ __device__ Translation& operator=(const Translation&) = delete;

	__host__ __device__ bool Hit(const Ray3& ray, float tMin, float tMax, HitRecord& hRec) const override;

protected:
	Hittable* hittable{ nullptr };
	Hittable** hittable_d{ nullptr };
	Vec3 offset;

	__host__ void constructEnvironment();
	__host__ void destroyEnvironment();
};

#endif