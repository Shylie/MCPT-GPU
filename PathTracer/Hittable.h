#ifndef HITTABLE_H
#define HITTABLE_H

#include "Common.h"
#include "Material.h"

#include <cuda_runtime.h>

struct HitRecord
{
	__host__ __device__ const Vec3& GetPoint() const { return _point; }
	__host__ __device__ void SetPoint(const Vec3& value) { _point = value; }

	__host__ __device__ const Vec3& GetNormal() const { return _normal; }
	__host__ __device__ void SetNormal(const Vec3& value) { _normal = value.Normalized(); }

	__host__ __device__ float GetT() const { return _t; }
	__host__ __device__ void SetT(float value) { _t = value; }

	__host__ __device__ Material** GetMaterial() const { return _mat_d; }
	__host__ __device__ void SetMaterial(Material** value) { _mat_d = value; }

	__host__ Material* GetMaterialHost() const { return _mat; }
	__host__ void SetMaterialHost(Material* value)
	{ 
		_mat = value;
		if (_mat != nullptr) _mat_d = _mat->GetPtrGPU();
	}

protected:
	Vec3 _point, _normal;
	float _t;
	Material** _mat_d{ nullptr };
	Material* _mat{ nullptr };
};

class API Hittable
{
public:
	__host__ __device__ virtual ~Hittable() { }

	__host__ __device__ Hittable(const Hittable&) = delete;
	__host__ __device__ Hittable& operator=(const Hittable&) = delete;

	__host__ __device__ virtual bool Hit(const Ray3& ray, float tMin, float tMax, HitRecord& hRec) const = 0;

	__host__ Hittable** GetPtrGPU() const { return this_d; }

protected:
	Hittable** this_d{ nullptr };

	__host__ __device__ Hittable() { }
};

#endif