#include "NoiseTexture.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void constructEnvironmentGPU_NoiseTexture(Texture** this_d, int tiles, Vec3* randv_d, int* permX_d, int* permY_d, int* permZ_d)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		(*this_d) = new NoiseTexture(tiles, randv_d, permX_d, permY_d, permZ_d);
	}
}

__global__ void destroyEnvironmentGPU_NoiseTexture(Texture** this_d)
{
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		delete (*this_d);
	}
}

void permute(int* p, int n)
{
	for (int i = n - 1; i >= 0; i--)
	{
		int target = int(float(rand()) / float(RAND_MAX) * (i + 1));
		int tmp = p[i];
		p[i] = p[target];
		p[target] = tmp;
	}
}

__host__ __device__ inline float perlinInterp(Vec3 c[2][2][2], float u, float v, float w)
{
	float uu = u * u * (3.0f - 2.0f * u);
	float vv = v * v * (3.0f - 2.0f * v);
	float ww = w * w * (3.0f - 2.0f * w);
	float accum = 0.0f;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			for (int k = 0; k < 2; k++)
			{
				Vec3 weightVec(u - i, v - j, w - k);
				accum += (i * uu + (1 - i) * (1 - uu)) * (j * vv + (1 - j) * (1 - vv)) * (k * ww + (1 - k) * (1 - ww)) * Vec3::Dot(c[i][j][k], weightVec);
			}
		}
	}
	return accum;
}

NoiseTexture::NoiseTexture(int tiles) : tiles(tiles), randv(new Vec3[tiles]), permX(new int[tiles]), permY(new int[tiles]), permZ(new int[tiles])
{
#ifndef __CUDA_ARCH__
	constructEnvironment();
#endif
}

__device__ NoiseTexture::NoiseTexture(int tiles, Vec3* rand_d, int* permX_d, int* permY_d, int* permZ_d) : tiles(tiles), randv_d(rand_d), permX_d(permX_d), permY_d(permY_d), permZ_d(permZ_d)
{
}

NoiseTexture::~NoiseTexture()
{
#ifndef __CUDA_ARCH__
	destroyEnvironment();
#endif
}

Vec3 NoiseTexture::Value(unsigned int* seed, float u, float v, const Vec3& pos) const
{
	return Vec3(1.0f) * Turb(seed, u, v, 4.0f * pos, 7);
}

float NoiseTexture::Noise(unsigned int* seed, float u, float v, const Vec3& pos) const
{
	float _u = pos.X - floor(pos.X);
	float _v = pos.Y - floor(pos.Y);
	float _w = pos.Z - floor(pos.Z);
	int i = floor(pos.X);
	int j = floor(pos.Y);
	int k = floor(pos.Z);
	Vec3 c[2][2][2];
	for (int di = 0; di < 2; di++)
	{
		for (int dj = 0; dj < 2; dj++)
		{
			for (int dk = 0; dk < 2; dk++)
			{
#ifdef __CUDA_ARCH__
				c[di][dj][dk] = randv_d[permX_d[(i + di) & (tiles - 1)] ^ permY_d[(j + dj) & (tiles - 1)] ^ permZ_d[(k + dk) & (tiles - 1)]];
#else
				c[di][dj][dk] = randv[permX[(i + di) & (tiles - 1)] ^ permY[(j + dj) & (tiles - 1)] ^ permZ[(k + dk) & (tiles - 1)]];
#endif
			}
		}
	}
	return perlinInterp(c, _u, _v, _w);
}

float NoiseTexture::Turb(unsigned int* seed, float u, float v, const Vec3& pos, int depth) const
{
	float accum = 0.0f;
	Vec3 tempP = pos;
	float weight = 1.0f;
	for (int i = 0; i < depth; i++)
	{
		accum += weight * Noise(seed, u, v, tempP);
		weight *= 0.5f;
		tempP = tempP * 2.0f;
	}
	return abs(accum);
}

void NoiseTexture::constructEnvironment()
{
	for (int i = 0; i < tiles; i++)
	{
		float x = float(rand()) / RAND_MAX;
		float y = float(rand()) / RAND_MAX;
		float z = float(rand()) / RAND_MAX;
		randv[i] = Vec3(-1.0f + 2.0f * x, -1.0f + 2.0f * y, -1.0f + 2.0f * z).Normalized();
		permX[i] = i;
		permY[i] = i;
		permZ[i] = i;
	}
	permute(permX, tiles);
	permute(permY, tiles);
	permute(permZ, tiles);

	cudaMalloc(&this_d, sizeof(Texture**));
	cudaMalloc(&randv_d, tiles * sizeof(Vec3));
	cudaMalloc(&permX_d, tiles * sizeof(int));
	cudaMalloc(&permY_d, tiles * sizeof(int));
	cudaMalloc(&permZ_d, tiles * sizeof(int));
	cudaMemcpy(randv_d, randv, tiles * sizeof(Vec3), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(permX_d, permX, tiles * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(permY_d, permY, tiles * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(permZ_d, permZ, tiles * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);
	constructEnvironmentGPU_NoiseTexture<<<1, 1>>>(this_d, tiles, randv_d, permX_d, permY_d, permZ_d);
	cudaDeviceSynchronize();
}

void NoiseTexture::destroyEnvironment()
{
	destroyEnvironmentGPU_NoiseTexture<<<1, 1>>>(this_d);
	cudaFree(this_d);
	cudaFree(randv_d);
	cudaFree(permX_d);
	cudaFree(permY_d);
	cudaFree(permZ_d);
	cudaDeviceSynchronize();

	delete[] randv, permX, permY, permZ;
}