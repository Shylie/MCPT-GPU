#ifndef LAMBERTIAN_H
#define LAMBERTIAN_H

#include "Material.h"
#include "Texture.h"

class API Lambertian : public Material
{
public:
	__host__ Lambertian(Texture* texture);
	__device__ Lambertian(Texture** texture_d);
	__host__ __device__ ~Lambertian();

	__host__ __device__ Lambertian(const Lambertian&) = delete;
	__host__ __device__ Lambertian& operator=(const Lambertian&) = delete;

	__host__ __device__ virtual bool Scatter(unsigned int* seed, Ray3& ray, const HitRecord& hRec, Vec3& attenuation) const override;

protected:
	Texture* texture{ nullptr };
	Texture** texture_d{ nullptr };

	__host__ void constructEnvironment();
	__host__ void destroyEnvironment();
};

#endif