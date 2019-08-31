#include "PlaneHittable.h"

__host__ __device__ bool PlaneHittable::Hit(const Ray3& ray, float tMin, float tMax, HitRecord& hRec) const
{
	float temp;
	Vec3 point;
	switch (alignment)
	{
	case (X | Y):
		temp = (k - ray.Origin().Z) / ray.Direction().Z;
		if (temp < tMin || temp > tMax) return false;
		point = ray.PointAt(temp);
		if (!Hit(point.X, point.Y)) return false;
		hRec.SetT(temp);
		hRec.SetPoint(point);
		if (autoNormal && ray.Direction().Z >= 0.0f)
		{
			if (ray.Direction().Z < 0.0f)
			{
				hRec.SetNormal(Vec3(0.0f, 0.0f, 1.0f));
			}
			else
			{
				hRec.SetNormal(Vec3(0.0f, 0.0f, -1.0f));
			}
		}
		else
		{
			if (invertNormal)
			{
				hRec.SetNormal(Vec3(0.0f, 0.0f, -1.0f));
			}
			else
			{
				hRec.SetNormal(Vec3(0.0f, 0.0f, 1.0f));
			}
		}
		hRec.SetMaterial(mat_d);
		return true;

	case (X | Z):
		temp = (k - ray.Origin().Y) / ray.Direction().Y;
		if (temp < tMin || temp > tMax) return false;
		point = ray.PointAt(temp);
		if (!Hit(point.X, point.Z)) return false;
		hRec.SetT(temp);
		hRec.SetPoint(point);
		if (autoNormal)
		{
			if (ray.Direction().Y < 0.0f)
			{
				hRec.SetNormal(Vec3(0.0f, 1.0f, 0.0f));
			}
			else
			{
				hRec.SetNormal(Vec3(0.0f, -1.0f, 0.0f));
			}
		}
		else
		{
			if (invertNormal)
			{
				hRec.SetNormal(Vec3(0.0f, -1.0f, 0.0f));
			}
			else
			{
				hRec.SetNormal(Vec3(0.0f, 1.0f, 0.0f));
			}
		}
		hRec.SetMaterial(mat_d);
		return true;

	case (Y | Z):
		temp = (k - ray.Origin().X) / ray.Direction().X;
		if (temp < tMin || temp > tMax) return false;
		point = ray.PointAt(temp);
		if (!Hit(point.Y, point.Z)) return false;
		hRec.SetT(temp);
		hRec.SetPoint(point);
		if (autoNormal)
		{
			if (ray.Direction().X < 0.0f)
			{
				hRec.SetNormal(Vec3(1.0f, 0.0f, 0.0f));
			}
			else
			{
				hRec.SetNormal(Vec3(-1.0f, 0.0f, 0.0f));
			}
		}
		else
		{
			if (invertNormal)
			{
				hRec.SetNormal(Vec3(-1.0f, 0.0f, 0.0f));
			}
			else
			{
				hRec.SetNormal(Vec3(1.0f, 0.0f, 0.0f));
			}
		}
		hRec.SetMaterial(mat_d);
		return true;

	default:
		return false;
	}
}