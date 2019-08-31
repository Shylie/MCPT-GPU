#include "SDFHittable.h"

constexpr float SDF_HITTABLE_NORMAL_SAMPLE_EPSILON = 0.00001f;
constexpr float SDF_HITTABLE_HIT_TEST_EPSILON = 0.001f;
constexpr float SDF_HITTABLE_MAX_HIT_TEST_ITERATIONS = 10000;

__host__ __device__ bool SDFHittable::Hit(const Ray3& ray, float tMin, float tMax, HitRecord& hRec) const
{
	float depth = tMin, dist = 0.0f;
	int iteration = 0;
	do
	{
		dist = Distance(ray.PointAt(depth));

		if (abs(dist) < SDF_HITTABLE_HIT_TEST_EPSILON)
		{
			hRec.SetT(depth);
			hRec.SetPoint(ray.PointAt(depth));
			hRec.SetNormal(Normal(hRec.GetPoint()));
			hRec.SetMaterial(MaterialAt(hRec.GetPoint()));
			return true;
		}

		depth += abs(dist / 2.5f);
	}
	while (iteration++ < SDF_HITTABLE_MAX_HIT_TEST_ITERATIONS && depth < tMax);
	return false;
}

__host__ __device__ Material** SDFHittable::MaterialAt(const Vec3& point) const
{
	return nullptr;
}

__host__ __device__ Vec3 SDFHittable::Normal(const Vec3& point) const
{
	float dx = Distance(point + Vec3(SDF_HITTABLE_HIT_TEST_EPSILON, 0.0f, 0.0f)) - Distance(point - Vec3(SDF_HITTABLE_HIT_TEST_EPSILON, 0.0f, 0.0f));
	float dy = Distance(point + Vec3(0.0f, SDF_HITTABLE_HIT_TEST_EPSILON, 0.0f)) - Distance(point - Vec3(0.0f, SDF_HITTABLE_HIT_TEST_EPSILON, 0.0f));
	float dz = Distance(point + Vec3(0.0f, 0.0f, SDF_HITTABLE_HIT_TEST_EPSILON)) - Distance(point - Vec3(0.0f, 0.0f, SDF_HITTABLE_HIT_TEST_EPSILON));
	return Vec3(dx, dy, dz).Normalized();
}