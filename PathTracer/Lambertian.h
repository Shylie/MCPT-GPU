#ifndef LAMBERTIAN_H
#define LAMBERTIAN_H

#include "Material.h"

class API Lambertian : public Material
{
public:
	__host__ __device__ Lambertian(Vec3 albedo);
	__host__ __device__ ~Lambertian();

	__host__ __device__ Lambertian(const Lambertian&) = delete;
	__host__ __device__ Lambertian& operator=(const Lambertian&) = delete;

	__host__ __device__ virtual bool Scatter(unsigned int* seed, Ray3& ray, const Vec3& point, const Vec3& normal, Vec3& attenuation) const override;

protected:
	Vec3 albedo;

	__host__ void constructEnvironment();
	__host__ void destroyEnvironment();
};

#endif