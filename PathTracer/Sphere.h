#ifndef SPHERE_H
#define SPHERE_H

#include "Hittable.h"

class API Sphere : public Hittable
{
public:
	__host__ Sphere(Vec3 center, float radius, Material* mat);
	__device__ Sphere(Vec3 center, float radius, Material** mat_d);
	__host__ __device__ ~Sphere();

	__host__ __device__ Sphere(const Sphere&) = delete;
	__host__ __device__ Sphere& operator=(const Sphere&) = delete;

	__host__ __device__ bool Hit(const Ray3& ray, float tMin, float tMax, HitRecord& hRec) const override;

protected:
	Vec3 center;
	float radius;
	Material* mat{ nullptr };
	Material** mat_d{ nullptr };

	__host__ void constructEnvironment();
	__host__ void destroyEnvironment();
};

#endif