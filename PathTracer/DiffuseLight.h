#ifndef DIFFUSE_LIGHT_H
#define DIFFUSE_LIGHT_H

#include "Material.h"
#include "Texture.h"

class API DiffuseLight : public Material
{
public:
	__host__ DiffuseLight(Texture* texture);
	__device__ DiffuseLight(Texture** texture_d);
	__host__ __device__ ~DiffuseLight();

	__host__ __device__ DiffuseLight(const DiffuseLight&) = delete;
	__host__ __device__ DiffuseLight& operator=(const DiffuseLight&) = delete;

	__host__ __device__ virtual Vec3 Emit(unsigned int* seed, const HitRecord& hRec) const override;

protected:
	Texture* texture{ nullptr };
	Texture** texture_d{ nullptr };

	__host__ void constructEnvironment();
	__host__ void destroyEnvironment();
};

#endif