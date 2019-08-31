#ifndef DIELETRIC_H
#define DIELETRIC_H

#include "Material.h"

class API Dieletric : public Material
{
public:
	__host__ __device__ Dieletric(Vec3 albedo, float refractiveIndex);
	__host__ __device__ ~Dieletric();

	__host__ __device__ Dieletric(const Dieletric&) = delete;
	__host__ __device__ Dieletric& operator=(const Dieletric&) = delete;

	__host__ __device__ virtual bool Scatter(unsigned int* seed, Ray3& ray, const Vec3& point, const Vec3& normal, Vec3& attenuation) const override;

protected:
	Vec3 albedo;
	float refractiveIndex;

	__host__ __device__ Vec3 Reflect(Vec3 v, Vec3 n) const;
	__host__ __device__ bool Refract(Vec3 v, Vec3 n, float niOverNt, Vec3& refracted) const;
	__host__ __device__ float Schlick(float cosine) const;

	__host__ void constructEnvironment();
	__host__ void destroyEnvironment();
};

#endif