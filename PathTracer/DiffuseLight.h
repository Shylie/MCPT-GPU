#ifndef DIFFUSE_LIGHT_H
#define DIFFUSE_LIGHT_H

#include "Material.h"

class API DiffuseLight : public Material
{
public:
	__host__ __device__ DiffuseLight(Vec3 color);
	__host__ __device__ ~DiffuseLight();

	__host__ __device__ DiffuseLight(const DiffuseLight&) = delete;
	__host__ __device__ DiffuseLight& operator=(const DiffuseLight&) = delete;

	__host__ __device__ virtual Vec3 Emit() const override;

protected:
	Vec3 color;

	__host__ void constructEnvironment();
	__host__ void destroyEnvironment();
};

#endif