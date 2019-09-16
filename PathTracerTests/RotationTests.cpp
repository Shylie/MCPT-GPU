#include <CppUnitTest.h>
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

#include "..\\PathTracer\Common.h"
#include "..\\PathTracer\RectangularPlane.h"
#include "..\\PathTracer\Rotation.h"

namespace PathTracerTests
{
	TEST_CLASS(RotationTests)
	{
	public:
		TEST_METHOD(RotationCtor)
		{
			Hittable* plane = new RectangularPlane(-1.0f, 1.0f, -1.0f, 1.0f, 0.0f, (Alignment)(X | Z), true, false, nullptr);
			Hittable* rotated = new Rotation(3.1415926f / 4.0f, Y, plane);
			
			Assert::IsNotNull(rotated->GetPtrGPU());

			delete plane;
			delete rotated;
		}

		TEST_METHOD(RotationX)
		{
			const float expectedT = 1.0f;
			const float expectedPointX = 0.0f;
			const float expectedPointY = -0.5f;
			const float expectedPointZ = 0.5f;
			const float expectedNormalX = -1.0f;
			const float expectedNormalY = 0.0f;
			const float expectedNormalZ = 0.0f;

			Hittable* plane = new RectangularPlane(-1.0f, 1.0f, -1.0f, 1.0f, 0.0f, (Alignment)(Y | Z), true, false, nullptr);
			Hittable* rotated = new Rotation(3.1415926f / 4.0f, X, plane);

			Ray3 rayMiss(Vec3(-1.0f, -0.75f, 0.75f), Vec3(1.0f, 0.0f, 0.0f));
			Ray3 rayHit(Vec3(-1.0f, -0.5f, 0.5f), Vec3(1.0f, 0.0f, 0.0f));

			HitRecord hRecMiss;
			HitRecord hRecHit;

			bool hitMiss = rotated->Hit(rayMiss, 0.001f, FLT_MAX, hRecMiss);
			bool hitHit = rotated->Hit(rayHit, 0.001f, FLT_MAX, hRecHit);

			Assert::IsFalse(hitMiss, L"Miss test hit");
			Assert::IsTrue(hitHit, L"Hit test missed");

			Assert::AreEqual(expectedT, hRecHit.GetT(), 0.00001f, L"T value");
			Assert::AreEqual(expectedPointX, hRecHit.GetPoint().X, 0.00001f, L"Point X value");
			Assert::AreEqual(expectedPointY, hRecHit.GetPoint().Y, 0.00001f, L"Point Y value");
			Assert::AreEqual(expectedPointZ, hRecHit.GetPoint().Z, 0.00001f, L"Point Z value");
			Assert::AreEqual(expectedNormalX, hRecHit.GetNormal().X, 0.00001f, L"Normal X value");
			Assert::AreEqual(expectedNormalY, hRecHit.GetNormal().Y, 0.00001f, L"Normal Y value");
			Assert::AreEqual(expectedNormalZ, hRecHit.GetNormal().Z, 0.00001f, L"Normal Z value");

			delete plane;
			delete rotated;
		}

		TEST_METHOD(RotationY)
		{
			const float expectedT = 1.0f;
			const float expectedPointX = -0.5f;
			const float expectedPointY = 0.0f;
			const float expectedPointZ = 0.5f;
			const float expectedNormalX = 0.0f;
			const float expectedNormalY = -1.0f;
			const float expectedNormalZ = 0.0f;

			Hittable* plane = new RectangularPlane(-1.0f, 1.0f, -1.0f, 1.0f, 0.0f, (Alignment)(X | Z), true, false, nullptr);
			Hittable* rotated = new Rotation(3.1415926f / 4.0f, Y, plane);

			Ray3 rayMiss(Vec3(-0.75f, -1.0f, 0.75f), Vec3(0.0f, 1.0f, 0.0f));
			Ray3 rayHit(Vec3(-0.5f, -1.0f, 0.5f), Vec3(0.0f, 1.0f, 0.0f));

			HitRecord hRecMiss;
			HitRecord hRecHit;

			bool hitMiss = rotated->Hit(rayMiss, 0.001f, FLT_MAX, hRecMiss);
			bool hitHit = rotated->Hit(rayHit, 0.001f, FLT_MAX, hRecHit);

			Assert::IsFalse(hitMiss, L"Miss test hit");
			Assert::IsTrue(hitHit, L"Hit test missed");

			Assert::AreEqual(expectedT, hRecHit.GetT(), 0.00001f, L"T value");
			Assert::AreEqual(expectedPointX, hRecHit.GetPoint().X, 0.00001f, L"Point X value");
			Assert::AreEqual(expectedPointY, hRecHit.GetPoint().Y, 0.00001f, L"Point Y value");
			Assert::AreEqual(expectedPointZ, hRecHit.GetPoint().Z, 0.00001f, L"Point Z value");
			Assert::AreEqual(expectedNormalX, hRecHit.GetNormal().X, 0.00001f, L"Normal X value");
			Assert::AreEqual(expectedNormalY, hRecHit.GetNormal().Y, 0.00001f, L"Normal Y value");
			Assert::AreEqual(expectedNormalZ, hRecHit.GetNormal().Z, 0.00001f, L"Normal Z value");

			delete plane;
			delete rotated;
		}

		TEST_METHOD(RotationZ)
		{
			const float expectedT = 1.0f;
			const float expectedPointX = -0.5f;
			const float expectedPointY = 0.5f;
			const float expectedPointZ = 0.0f;
			const float expectedNormalX = 0.0f;
			const float expectedNormalY = 0.0f;
			const float expectedNormalZ = -1.0f;

			Hittable* plane = new RectangularPlane(-1.0f, 1.0f, -1.0f, 1.0f, 0.0f, (Alignment)(X | Y), true, false, nullptr);
			Hittable* rotated = new Rotation(3.1415926f / 4.0f, Z, plane);

			Ray3 rayMiss(Vec3(0.75f, 0.75f, -1.0f), Vec3(0.0f, 0.0f, 1.0f));
			Ray3 rayHit(Vec3(-0.5f, 0.5f, -1.0f), Vec3(0.0f, 0.0f, 1.0f));

			HitRecord hRecMiss;
			HitRecord hRecHit;

			bool hitMiss = rotated->Hit(rayMiss, 0.001f, FLT_MAX, hRecMiss);
			bool hitHit = rotated->Hit(rayHit, 0.001f, FLT_MAX, hRecHit);

			Assert::IsFalse(hitMiss, L"Miss test hit");
			Assert::IsTrue(hitHit, L"Hit test missed");

			Assert::AreEqual(expectedT, hRecHit.GetT(), 0.00001f, L"T value");
			Assert::AreEqual(expectedPointX, hRecHit.GetPoint().X, 0.00001f, L"Point X value");
			Assert::AreEqual(expectedPointY, hRecHit.GetPoint().Y, 0.00001f, L"Point Y value");
			Assert::AreEqual(expectedPointZ, hRecHit.GetPoint().Z, 0.00001f, L"Point Z value");
			Assert::AreEqual(expectedNormalX, hRecHit.GetNormal().X, 0.00001f, L"Normal X value");
			Assert::AreEqual(expectedNormalY, hRecHit.GetNormal().Y, 0.00001f, L"Normal Y value");
			Assert::AreEqual(expectedNormalZ, hRecHit.GetNormal().Z, 0.00001f, L"Normal Z value");

			delete plane;
			delete rotated;
		}
	};
}