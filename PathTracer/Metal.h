#ifndef METAL_H
#define METAL_H

#include "Material.h"

class API Metal : public Material
{
public:
	__host__ __device__ Metal(Vec3 albedo, float fuzz);
	__host__ __device__ ~Metal();

	__host__ __device__ Metal(const Metal&) = delete;
	__host__ __device__ Metal& operator=(const Metal&) = delete;

	__host__ __device__ virtual bool Scatter(unsigned int* seed, Ray3& ray, const Vec3& point, const Vec3& normal, Vec3& attenuation) const override;

protected:
	Vec3 albedo;
	float fuzz;

	__host__ __device__ Vec3 Reflect(Vec3 v, Vec3 n) const;

	__host__ void constructEnvironment();
	__host__ void destroyEnvironment();
};

#endif