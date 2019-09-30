#ifndef HIT_RECORD_H
#define HIT_RECORD_H

#include "Common.h"

#include <cuda_runtime.h>

class Material;

struct HitRecord
{
	__host__ __device__ const Vec3& GetPoint() const { return _point; }
	__host__ __device__ void SetPoint(const Vec3& value) { _point = value; }

	__host__ __device__ const Vec3& GetNormal() const { return _normal; }
	__host__ __device__ void SetNormal(const Vec3& value) { _normal = value.Normalized(); }

	__host__ __device__ float GetT() const { return _t; }
	__host__ __device__ void SetT(float value) { _t = value; }

	__host__ __device__ float GetU() const { return _u; }
	__host__ __device__ void SetU(float value) { _u = (value < 0.0f) ? 0.0f : (value > 1.0f ? 1.0f : value); }

	__host__ __device__ float GetV() const { return _v; }
	__host__ __device__ void SetV(float value) { _v = (value < 0.0f) ? 0.0f : (value > 1.0f ? 1.0f : value); }

	__host__ __device__ Material** GetMaterial() const { return _mat_d; }
	__host__ __device__ void SetMaterial(Material** value) { _mat_d = value; }

	__host__ Material* GetMaterialHost() const { return _mat; }
	__host__ void SetMaterialHost(Material* value);

protected:
	Vec3 _point, _normal;
	float _t, _u, _v;
	Material** _mat_d{ nullptr };
	Material* _mat{ nullptr };
};

#endif