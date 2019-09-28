#ifndef CHECKERBOARD_TEXTURE_H
#define CHECKERBOARD_TEXTURE_H

#include "Texture.h"

class API CheckerboardTexture : public Texture
{
public:
	__host__ CheckerboardTexture(Texture* a, Texture* b, Vec3 offset, Vec3 frequency);
	__device__ CheckerboardTexture(Texture** a_d, Texture** b_d, Vec3 offset, Vec3 frequency);

	__host__ __device__ CheckerboardTexture(const CheckerboardTexture&) = delete;
	__host__ __device__ CheckerboardTexture& operator=(const CheckerboardTexture&) = delete;

	__host__ __device__ Vec3 Value(unsigned int* seed, const Vec3& pos) const override;

protected:
	Texture* a{ nullptr };
	Texture* b{ nullptr };
	Texture** a_d{ nullptr };
	Texture** b_d{ nullptr };

	Vec3 offset, frequency;

	__host__ void constructEnvironment();
	__host__ void destroyEnvironment();
};

#endif