#ifndef TRIANGULAR_PLANE_H
#define TRIANGULAR_PLANE_H

#include "PlaneHittable.h"

class API TriangularPlane : public PlaneHittable
{
public:
	__host__ __device__ TriangularPlane(float a1, float b1, float a2, float b2, float a3, float b3, float k, Alignment alignment, bool autoNormal, bool invertNormal, Material** mat_d);
	__host__ __device__ ~TriangularPlane();

	__host__ __device__ TriangularPlane(const TriangularPlane&) = delete;
	__host__ __device__ TriangularPlane& operator=(const TriangularPlane&) = delete;

protected:
	__host__ __device__ bool Hit(float a, float b) const override;

	float a1, a2, a3, b1, b2, b3;

	__host__ void constructEnvironment();
	__host__ void destroyEnvironment();
};

#endif