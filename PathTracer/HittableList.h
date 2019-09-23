#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "Hittable.h"

class API HittableList : public Hittable
{
public:
	__host__ HittableList(int numHittables, Hittable** hittables);
	__device__ HittableList(int numHittables, Hittable*** hittables_d);
	__host__ __device__ ~HittableList();

	__host__ __device__ HittableList(const HittableList&) = delete;
	__host__ __device__ HittableList& operator=(const HittableList&) = delete;

	__host__ __device__ bool Hit(const Ray3& ray, float tMin, float tMax, HitRecord& hRec) const override;

protected:
	int numHittables;
	Hittable** hittables{ nullptr };
	Hittable*** hittables_d{ nullptr };
	Hittable*** device_hittables_d{ nullptr };

	__host__ void constructEnvironment();
	__host__ void destroyEnvironment();
};

#endif