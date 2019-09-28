#ifndef PLANE_HITTABLE_H
#define PLANE_HITTABLE_H

#include "Hittable.h"
#include "Material.h"

class API PlaneHittable : public Hittable
{
public:
	__host__ PlaneHittable(float k, Alignment alignment, bool autoNormal, bool invertNormal, Material* mat) : k(k), alignment(alignment), autoNormal(autoNormal), invertNormal(invertNormal), mat(mat), mat_d(mat != nullptr ? mat->GetPtrGPU() : nullptr) { }
	__device__ PlaneHittable(float k, Alignment alignment, bool autoNormal, bool invertNormal, Material** mat_d) : k(k), alignment(alignment), autoNormal(autoNormal), invertNormal(invertNormal), mat_d(mat_d) { }

	__host__ __device__ PlaneHittable(const PlaneHittable&) = delete;
	__host__ __device__ PlaneHittable& operator=(const PlaneHittable&) = delete;

	__host__ __device__ bool Hit(const Ray3& ray, float tMin, float tMax, HitRecord& hRec) const override;
	__host__ __device__ virtual bool Hit(float a, float b) const = 0;

protected:

	Material** mat_d{ nullptr };
	Material* mat{ nullptr };
	Alignment alignment;
	float k;
	bool autoNormal, invertNormal;
};

#endif