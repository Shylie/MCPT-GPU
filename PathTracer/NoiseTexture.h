#ifndef NOISE_TEXTURE_H
#define NOISE_TEXTURE_H

#include "Texture.h"

class API NoiseTexture : public Texture
{
public:
	__host__ NoiseTexture(int tiles);
	__device__ NoiseTexture(int tiles, Vec3* randv_d, int* permX_d, int* permY_d, int* permZ_d);
	__host__ __device__ ~NoiseTexture();

	__host__ __device__ NoiseTexture(const NoiseTexture&) = delete;
	__host__ __device__ NoiseTexture& operator=(const NoiseTexture&) = delete;

	__host__ __device__ Vec3 Value(unsigned int* seed, float u, float v, const Vec3& pos) const override;

protected:
	int tiles;
	Vec3* randv{ nullptr };
	int* permX{ nullptr };
	int* permY{ nullptr };
	int* permZ{ nullptr };
	Vec3* randv_d{ nullptr };
	int* permX_d{ nullptr };
	int* permY_d{ nullptr };
	int* permZ_d{ nullptr };

	__host__ __device__ float Noise(unsigned int* seed, float u, float v, const Vec3& pos) const;
	__host__ __device__ float Turb(unsigned int* seed, float u, float v, const Vec3& pos, int depth) const;

	__host__ void constructEnvironment();
	__host__ void destroyEnvironment();
};

#endif