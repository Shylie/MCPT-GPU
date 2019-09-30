#ifndef MATERIAL_H
#define MATERIAL_H

#include "Common.h"
#include "HitRecord.h"

#include <cuda_runtime.h>

class API Material
{
public:
	__host__ __device__ virtual ~Material() { }

	__host__ __device__ Material(const Material&) = delete;
	__host__ __device__ Material& operator=(const Material&) = delete;

	__host__ __device__ virtual bool Scatter(unsigned int* seed, Ray3& ray, const HitRecord& hRec, Vec3& attenuation) const { return false; }
	__host__ __device__ virtual Vec3 Emit(unsigned int* seed, const HitRecord& hRec) const { return Vec3(0.0f); }

	__host__ Material** GetPtrGPU() const { return this_d; }

protected:
	Material** this_d{ nullptr };
	
	__host__ __device__ Material() { }
};

#endif