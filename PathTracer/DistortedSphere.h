#ifndef DISTORTED_SPHERE_H
#define DISTORTED_SPHERE_H

#include "SDFHittable.h"

class API DistortedSphere : public SDFHittable
{
public:
	__host__ __device__ DistortedSphere(Vec3 center, float radius, float frequency, float amplitude, Material** mat_d);
	__host__ __device__ ~DistortedSphere();

	__host__ __device__ DistortedSphere(const DistortedSphere&) = delete;
	__host__ __device__ DistortedSphere& operator=(const DistortedSphere&) = delete;

protected:
	__host__ __device__ float Distance(const Vec3& point) const override;
	__host__ __device__ Material** MaterialAt(const Vec3& point) const override;

	Vec3 center;
	float radius, frequency, amplitude;
	Material** mat_d;

	__host__ void constructEnvironment();
	__host__ void destroyEnvironment();
};

#endif