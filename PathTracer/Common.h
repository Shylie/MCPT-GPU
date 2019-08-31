#ifndef COMMON_H
#define COMMON_H

#ifdef __cplusplus
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif

#include <host_defines.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>

__host__ __device__ inline float randXORShift(unsigned int* seed)
{
	*seed ^= (*seed << 13);
	*seed ^= (*seed >> 17);
	*seed ^= (*seed << 5);
	return *seed / float(UINT_MAX);
}

__host__ __device__ inline void wangHash(unsigned int* seed)
{
	*seed = (*seed ^ 61) ^ (*seed >> 16);
	*seed *= 9;
	*seed = *seed ^ (*seed >> 4);
	*seed *= 0x27d4eb2d;
	*seed = *seed ^ (*seed >> 15);
}

enum API Alignment
{
	None = 0,
	X = 1 << 0,
	Y = 1 << 1,
	Z = 1 << 2
};

struct API Vec3
{
	float X;
	float Y;
	float Z;

	__host__ __device__ Vec3() : X(0.0f), Y(0.0f), Z(0.0f) { }
	__host__ __device__ Vec3(float val) : X(val), Y(val), Z(val) { }
	__host__ __device__ Vec3(float x, float y, float z) : X(x), Y(y), Z(z) { }

	__host__ __device__ float Length() const { return sqrt(X * X + Y * Y + Z * Z); }
	__host__ __device__ float LengthSquared() const { return X * X + Y * Y + Z * Z; }

	__host__ __device__ Vec3 Normalized() const
	{
		float l = Length();
		return Vec3(X / l, Y / l, Z / l);
	}

	__host__ __device__ static float Dot(Vec3 a, Vec3 b) { return a.X * b.X + a.Y * b.Y + a.Z * b.Z; }
	__host__ __device__ static Vec3 Cross(Vec3 a, Vec3 b) { return Vec3(a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X); }
	__host__ __device__ static Vec3 RandomUnitVector(unsigned int* seed)
	{
		return Vec3(randXORShift(seed) * 2.0f - 1.0f, randXORShift(seed) * 2.0f - 1.0f, randXORShift(seed) * 2.0f - 1.0f).Normalized();
	}
};

__host__ __device__ inline Vec3 operator-(Vec3 vec) { return Vec3(-vec.X, -vec.Y, -vec.Z); }
__host__ __device__ inline Vec3 operator+(Vec3 a, Vec3 b) { return Vec3(a.X + b.X, a.Y + b.Y, a.Z + b.Z); }
__host__ __device__ inline Vec3 operator-(Vec3 a, Vec3 b) { return Vec3(a.X - b.X, a.Y - b.Y, a.Z - b.Z); }
__host__ __device__ inline Vec3 operator*(Vec3 a, Vec3 b) { return Vec3(a.X * b.X, a.Y * b.Y, a.Z * b.Z); }
__host__ __device__ inline Vec3 operator/(Vec3 a, Vec3 b) { return Vec3(a.X / b.X, a.Y / b.Y, a.Z / b.Z); }

__host__ __device__ inline Vec3 operator*(Vec3 vec, float scalar) { return Vec3(vec.X * scalar, vec.Y * scalar, vec.Z * scalar); }
__host__ __device__ inline Vec3 operator*(float scalar, Vec3 vec) { return vec * scalar; }
__host__ __device__ inline Vec3 operator/(Vec3 vec, float scalar) { return Vec3(vec.X / scalar, vec.Y / scalar, vec.Z / scalar); }
__host__ __device__ inline Vec3 operator/(float scalar, Vec3 vec) { return vec / scalar; }

struct API Ray3
{
	__host__ __device__ Ray3(Vec3 o, Vec3 d)
	{
		origin = o;
		direction = d.Normalized();
	}

	__host__ __device__ Vec3 PointAt(float t) const { return origin + t * direction; }

	__host__ __device__ Vec3 Origin() const { return origin; }
	__host__ __device__ Vec3 Direction() const { return direction; }

private:
	Vec3 origin, direction;
};

inline void saveImage(int width, int height, Vec3* colors, const char* fname)
{
	std::ofstream file;
	file.open(fname, std::ios::binary | std::ios::trunc);
	file << "P6" << std::endl;
	file << std::to_string(width) << ' ' << std::to_string(height) << std::endl;
	file << "255" << std::endl;

	for (int j = height - 1; j >= 0; j--)
	{
		for (int i = 0; i < width; i++)
		{
			float r = sqrt(colors[i + j * width].X);
			float g = sqrt(colors[i + j * width].Y);
			float b = sqrt(colors[i + j * width].Z);
			if (r > 1.0f) r = 1.0f;
			if (g > 1.0f) g = 1.0f;
			if (b > 1.0f) b = 1.0f;
			int8_t ir = int8_t(255.0f * r);
			int8_t ig = int8_t(255.0f * g);
			int8_t ib = int8_t(255.0f * b);
			file << ir << ig << ib;
		}
	}

	file.close();
}

struct Camera
{
	__host__ __device__ Camera(Vec3 lookFrom, Vec3 lookAt, Vec3 vup, float vfov, float aspect)
	{
		Vec3 u, v, w;
		float hh = tan(vfov / 2.0f);
		float hw = aspect * hh;
		origin = lookFrom;
		w = (lookFrom - lookAt).Normalized();
		u = Vec3::Cross(vup, w).Normalized();
		v = Vec3::Cross(w, u);
		llc = origin - hw * u - hh * v - w;
		horizontal = 2.0f * hw * u;
		vertical = 2.0f * hh * v;
	}

	__host__ __device__ inline Ray3 GetRay(float u, float v) const { return Ray3(origin, llc + u * horizontal + v * vertical - origin); }

private:
	Vec3 origin, llc, horizontal, vertical;
};

#endif