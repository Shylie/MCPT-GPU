#ifndef CONSTANT_TEXTURE_H
#define CONSTANT_TEXTURE_H

#include "Texture.h"

class API ConstantTexture : public Texture
{
public:
	__host__ __device__ ConstantTexture(Vec3 color);
	__host__ __device__ ~ConstantTexture();

	__host__ __device__ ConstantTexture(const ConstantTexture&) = delete;
	__host__ __device__ ConstantTexture& operator=(const ConstantTexture&) = delete;

	__host__ __device__ Vec3 Value(unsigned int* seed, float u, float v, const Vec3& pos) const override;

protected:
	Vec3 color;

	__host__ void constructEnvironment();
	__host__ void destroyEnvironment();
};

#endif