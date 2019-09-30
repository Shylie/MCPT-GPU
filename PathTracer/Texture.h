#ifndef TEXTURE_H
#define TEXTURE_H

#include "Common.h"

#include <cuda_runtime.h>

class API Texture
{
public:
	__host__ __device__ virtual ~Texture() { }

	__host__ __device__ Texture(const Texture&) = delete;
	__host__ __device__ Texture& operator=(const Texture&) = delete;

	__host__ __device__ virtual Vec3 Value(unsigned int* seed, float u, float v, const Vec3& pos) const = 0;

	__host__ Texture** GetPtrGPU() const { return this_d; }

protected:
	Texture** this_d{ nullptr };

	__host__ __device__ Texture() { }
};

#endif