#ifndef PLANE_HITTABLE_H
#define PLANE_HITTABLE_H

#include "Hittable.h"

class API PlaneHittable : public Hittable
{
public:
	__host__ __device__ PlaneHittable(float k, Alignment alignment, bool autoNormal, bool invertNormal, Material** mat_d) : k(k), alignment(alignment), autoNormal(autoNormal), invertNormal(invertNormal), mat_d(mat_d) { }

	__host__ __device__ PlaneHittable(const PlaneHittable&) = delete;
	__host__ __device__ PlaneHittable& operator=(const PlaneHittable&) = delete;

	__host__ __device__ bool Hit(const Ray3& ray, float tMin, float tMax, HitRecord& hRec) const override;

protected:
	__host__ __device__ virtual bool Hit(float a, float b) const = 0;

	Material** mat_d;
	Alignment alignment;
	float k;
	bool autoNormal, invertNormal;
};

#endif