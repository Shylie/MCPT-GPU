#include "Common.h"

#include "Hittable.h"
#include "SDFHittable.h"
#include "PlaneHittable.h"
#include "Material.h"

#include "Sphere.h"
#include "DistortedSphere.h"
#include "RectangularPlane.h"
#include "Translation.h"
#include "Rotation.h"
#include "HittableList.h"

#include "Lambertian.h"
#include "Metal.h"
#include "DiffuseLight.h"
#include "Dielectric.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <string>

__device__ bool getColor(unsigned int* seed, Ray3& ray, Vec3& attenuation, Vec3& emitted, Hittable** scene)
{
	HitRecord hRec;
	if ((*scene)->Hit(ray, 0.001f, 1e38f, hRec))
	{
		if (hRec.GetMaterial() != nullptr)
		{
			emitted = (*hRec.GetMaterial())->Emit();
			return (*hRec.GetMaterial())->Scatter(seed, ray, hRec.GetPoint(), hRec.GetNormal(), attenuation);
		}
	}
	attenuation = Vec3(0.0f);
	emitted = Vec3(0.0f);
	return false;
}

__global__ void renderGPU_All(Vec3* cols, int width, int height, int samples, Camera cam, Hittable** scene)
{
	for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < width; i += blockDim.x * gridDim.x)
	{
		for (int j = threadIdx.y + blockDim.y * blockIdx.y; j < height; j += blockDim.y * gridDim.y)
		{
			if (i + j * width < width * height)
			{
				unsigned int* seed = new unsigned int;
				*seed = (i + 1) ^ (j + 1) + (i + 1) * (j + 1);
				Vec3 avg;

				for (int n = 0; n < samples; n++)
				{
					wangHash(seed);

					float u = (i + randXORShift(seed) - 0.5f) / float(width), v = (j + randXORShift(seed) - 0.5f) / float(height);

					Ray3 ray = cam.GetRay(u, v);
					Vec3 attenuation, emitted, sum;
					Vec3 multiplier = Vec3(1.0f);

					int depth = 0;

					bool hit = false;
					do
					{
						hit = getColor(seed, ray, attenuation, emitted, scene);
						sum = sum + emitted * multiplier;
						multiplier = multiplier * attenuation;
					}
					while (hit && depth++ < 100);

					avg = avg + sum;
				}

				delete seed;

				cols[i + j * width] = avg / float(samples);
			}
		}
	}
}

__global__ void renderGPU_Chunk(Vec3* cols, int width, int height, int samples, Camera cam, Hittable** scene, int startX, int endX, int startY, int endY)
{
	for (int i = threadIdx.x + blockDim.x * blockIdx.x + startX; i < endX; i += blockDim.x * gridDim.x)
	{
		for (int j = threadIdx.y + blockDim.y * blockIdx.y + startY; j < endY; j += blockDim.y * gridDim.y)
		{
			if (i + j * width < width * height)
			{
				unsigned int* seed = new unsigned int;
				*seed = (i + 1) ^ (j + 1) + (i + 1) * (j + 1);
				Vec3 avg;

				for (int n = 0; n < samples; n++)
				{
					wangHash(seed);

					float u = (i + randXORShift(seed) - 0.5f) / float(width), v = (j + randXORShift(seed) - 0.5f) / float(height);

					Ray3 ray = cam.GetRay(u, v);
					Vec3 attenuation, emitted, sum;
					Vec3 multiplier = Vec3(1.0f);

					int depth = 0;

					bool hit = false;
					do
					{
						hit = getColor(seed, ray, attenuation, emitted, scene);
						sum = sum + emitted * multiplier;
						multiplier = multiplier * attenuation;
					} while (hit && depth++ < 100);

					avg = avg + sum;
				}

				delete seed;

				cols[(i - startX) + (j - startY) * (endX - startX)] = avg / float(samples);
			}
		}
	}
}

void render(int width, int height, int samples, const char* fname, Camera cam, Hittable** scene)
{
	Vec3* src;
	cudaMalloc(&src, width * height * sizeof(Vec3));

	Vec3* dst = new Vec3[width * height];

	dim3 blockSize = dim3(16, 16);
	dim3 numBlocks = dim3(16, 16);

	renderGPU_All<<<numBlocks, blockSize>>>(src, width, height, samples, cam, scene);

	cudaDeviceSynchronize();

	cudaMemcpy(dst, src, width * height * sizeof(Vec3), cudaMemcpyKind::cudaMemcpyDeviceToHost);
	cudaFree(src);

	saveImage(width, height, dst, fname);
}

void renderChunked(int width, int height, int samples, const char* fname, Camera cam, Hittable** scene)
{
	const int chunkSize = 64;

	Vec3* dst = new Vec3[width * height];

	dim3 blockSize = dim3(16, 16);
	dim3 numBlocks = dim3(4, 4);

	int processed = 0;

	for (int sx = 0; sx < width; sx += chunkSize)
	{
		for (int sy = 0; sy < height; sy += chunkSize)
		{
			int chunkWidth = chunkSize, chunkHeight = chunkSize;
			if (sx + chunkWidth > width) chunkWidth = width - sx;
			if (sy + chunkHeight > height) chunkHeight = height - sy;

			printf("\r%i / %i          ", processed, width * height);

			Vec3* src;
			cudaMalloc(&src, chunkWidth * chunkHeight * sizeof(Vec3));

			Vec3* tmpdst = new Vec3[chunkWidth * chunkHeight];

			renderGPU_Chunk<<<numBlocks, blockSize>>>(src, width, height, samples, cam, scene, sx, sx + chunkWidth, sy, sy + chunkWidth);

			cudaDeviceSynchronize();

			cudaMemcpy(tmpdst, src, chunkWidth * chunkHeight * sizeof(Vec3), cudaMemcpyKind::cudaMemcpyDeviceToHost);
			cudaFree(src);

			processed += chunkWidth * chunkHeight;

			for (int i = sx; i < sx + chunkWidth; i++)
			{
				for (int j = sy; j < sy + chunkHeight; j++)
				{
					dst[i + j * width] = tmpdst[(i - sx) + (j - sy) * (chunkWidth)];
				}
			}

			delete[] tmpdst;
		}
	}

	saveImage(width, height, dst, fname);
}

extern "C"
{
	API void RenderScene(int width, int height, int samples, const char* fname, float lfx, float lfy, float lfz, float lax, float lay, float laz, float upx, float upy, float upz, float vfov, float aspect, Hittable* scene)
	{
		Camera cam = Camera(Vec3(lfx, lfy, lfz), Vec3(lax, lay, laz), Vec3(upx, upy, upz), vfov, aspect);
		render(width, height, samples, fname, cam, scene->GetPtrGPU());
	}

	API void RenderSceneChunked(int width, int height, int samples, const char* fname, float lfx, float lfy, float lfz, float lax, float lay, float laz, float upx, float upy, float upz, float vfov, float aspect, Hittable* scene)
	{
		Camera cam = Camera(Vec3(lfx, lfy, lfz), Vec3(lax, lay, laz), Vec3(upx, upy, upz), vfov, aspect);
		renderChunked(width, height, samples, fname, cam, scene->GetPtrGPU());
	}

	API Hittable* ConstructHittableList(int numHittables, Hittable** hittables)
	{
		Hittable*** temp = new Hittable * *[numHittables];
		for (int i = 0; i < numHittables; i++)
		{
			temp[i] = hittables[i]->GetPtrGPU();
		}
		return new HittableList(numHittables, temp);
	}

	API Hittable* ConstructTranslation(float dx, float dy, float dz, Hittable* hittable)
	{
		return new Translation(Vec3(dx, dy, dz), hittable->GetPtrGPU());
	}

	API Hittable* ConstructRotation(float theta, Alignment alignment, Hittable* hittable)
	{
		return new Rotation(theta, alignment, hittable->GetPtrGPU());
	}

	API Hittable* ConstructSphere(float cx, float cy, float cz, float radius, Material* mat)
	{
		return new Sphere(Vec3(cx, cy, cz), radius, mat->GetPtrGPU());
	}

	API Hittable* ConstructRectangularPlane(float a1, float a2, float b1, float b2, float k, Alignment alignment, bool autoNormal, bool invertNormal, Material* mat)
	{
		return new RectangularPlane(a1, a2, b1, b2, k, alignment, autoNormal, invertNormal, mat->GetPtrGPU());
	}

	API Hittable* ConstructDistortedSphere(float cx, float cy, float cz, float radius, float frequency, float amplitude, Material* mat)
	{
		return new DistortedSphere(Vec3(cx, cy, cz), radius, frequency, amplitude, mat->GetPtrGPU());
	}

	API Material* ConstructLambertian(float r, float g, float b)
	{
		return new Lambertian(Vec3(r, g, b));
	}

	API Material* ConstructMetal(float r, float g, float b, float fuzz)
	{
		return new Metal(Vec3(r, g, b), fuzz);
	}

	API Material* ConstructDielectric(float r, float g, float b, float refIdx)
	{
		return new Dieletric(Vec3(r, g, b), refIdx);
	}

	API Material* ConstructDiffuseLight(float r, float g, float b)
	{
		return new DiffuseLight(Vec3(r, g, b));
	}

	API void DestroyHittable(Hittable* ptr)
	{
		delete ptr;
	}

	API void DestroyMaterial(Material* ptr)
	{
		delete ptr;
	}
}