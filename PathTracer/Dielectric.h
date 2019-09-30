#ifndef DIELETRIC_H
#define DIELETRIC_H

#include "Material.h"
#include "Texture.h"

class API Dieletric : public Material
{
public:
	__host__ Dieletric(Texture* texture, float refractiveIndex);
	__device__ Dieletric(Texture** texture_d, float refractiveIndex);
	__host__ __device__ ~Dieletric();

	__host__ __device__ Dieletric(const Dieletric&) = delete;
	__host__ __device__ Dieletric& operator=(const Dieletric&) = delete;

	__host__ __device__ virtual bool Scatter(unsigned int* seed, Ray3& ray, const HitRecord& hRec, Vec3& attenuation) const override;

protected:
	Texture* texture{ nullptr };
	Texture** texture_d{ nullptr };
	float refractiveIndex;

	__host__ __device__ Vec3 Reflect(Vec3 v, Vec3 n) const;
	__host__ __device__ bool Refract(Vec3 v, Vec3 n, float niOverNt, Vec3& refracted) const;
	__host__ __device__ float Schlick(float cosine) const;

	__host__ void constructEnvironment();
	__host__ void destroyEnvironment();
};

#endif