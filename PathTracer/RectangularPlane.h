#ifndef RECTANGULAR_PLANE_H
#define RECTANGULAR_PLANE_H

#include "PlaneHittable.h"

class API RectangularPlane : public PlaneHittable
{
public:
	__host__ __device__ RectangularPlane(float a1, float a2, float b1, float b2, float k, Alignment alignment, bool autoNormal, bool invertNormal, Material** mat_d);
	__host__ __device__ ~RectangularPlane();

	__host__ __device__ RectangularPlane(const RectangularPlane&) = delete;
	__host__ __device__ RectangularPlane& operator=(const RectangularPlane&) = delete;

protected:
	__host__ __device__ bool Hit(float a, float b) const override;

	float a1, a2, b1, b2;

	__host__ void constructEnvironment();
	__host__ void destroyEnvironment();
};

#endif