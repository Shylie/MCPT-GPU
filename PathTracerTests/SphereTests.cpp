#include <CppUnitTest.h>
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

#include "..\\PathTracer\Common.h"
#include "..\\PathTracer\Sphere.h"

namespace PathTracerTests
{
	TEST_CLASS(SphereTests)
	{
	public:
		TEST_METHOD(SphereCtor)
		{
			Hittable* sphere = new Sphere(Vec3(0.0f), 1.0f, nullptr);

			Assert::IsNotNull(sphere->GetPtrGPU(), L"Device ptr non-null");

			delete sphere;
		}

		TEST_METHOD(SphereHitFromOutside)
		{
			const float expectedT = 2.4641016151377545870548926830117f;
			const float expectedPointX = 2.0f - 0.57735026918962576450914878050196f;
			const float expectedPointY = 2.0f - 0.57735026918962576450914878050196f;
			const float expectedPointZ = 2.0f - 0.57735026918962576450914878050196f;
			const float expectedNormalX = -0.57735026918962576450914878050196f;
			const float expectedNormalY = -0.57735026918962576450914878050196f;
			const float expectedNormalZ = -0.57735026918962576450914878050196f;

			Hittable* sphere = new Sphere(Vec3(2.0f), 1.0f, nullptr);

			HitRecord hRec;

			Ray3 ray(Vec3(0.0f), Vec3(1.0f));

			bool hit = sphere->Hit(ray, 0.001f, FLT_MAX, hRec);

			Assert::IsTrue(hit, L"Hit sphere");
			Assert::AreEqual(expectedT, hRec.GetT(), L"T value");
			Assert::AreEqual(expectedPointX, hRec.GetPoint().X, 0.000001f, L"Point X value");
			Assert::AreEqual(expectedPointY, hRec.GetPoint().Y, 0.000001f, L"Point Y value");
			Assert::AreEqual(expectedPointZ, hRec.GetPoint().Z, 0.000001f, L"Point Z value");
			Assert::AreEqual(expectedNormalX, hRec.GetNormal().X, 0.000001f, L"Normal X value");
			Assert::AreEqual(expectedNormalY, hRec.GetNormal().Y, 0.000001f, L"Normal Y value");
			Assert::AreEqual(expectedNormalZ, hRec.GetNormal().Z, 0.000001f, L"Normal Z value");

			delete sphere;
		}

		TEST_METHOD(SphereMissFromOutside)
		{
			Hittable* sphere = new Sphere(Vec3(2.0f), 1.0f, nullptr);

			HitRecord hRec;

			Ray3 ray(Vec3(0.0f), Vec3(-1.0f));

			bool hit = sphere->Hit(ray, 0.001f, FLT_MAX, hRec);

			Assert::IsFalse(hit, L"Hit sphere");

			delete sphere;
		}

		TEST_METHOD(SphereHitFromInside)
		{
			const float expectedT = 1.0f;
			const float expectedPointX = 2.0f + 0.57735026918962576450914878050196f;
			const float expectedPointY = 2.0f + 0.57735026918962576450914878050196f;
			const float expectedPointZ = 2.0f + 0.57735026918962576450914878050196f;
			const float expectedNormalX = 0.57735026918962576450914878050196f;
			const float expectedNormalY = 0.57735026918962576450914878050196f;
			const float expectedNormalZ = 0.57735026918962576450914878050196f;

			Hittable* sphere = new Sphere(Vec3(2.0f), 1.0f, nullptr);

			HitRecord hRec;

			Ray3 ray(Vec3(2.0f), Vec3(1.0f));

			bool hit = sphere->Hit(ray, 0.001f, FLT_MAX, hRec);

			Assert::IsTrue(hit, L"Hit sphere");
			Assert::AreEqual(expectedT, hRec.GetT(), L"T value");
			Assert::AreEqual(expectedPointX, hRec.GetPoint().X, L"Point X value");
			Assert::AreEqual(expectedPointY, hRec.GetPoint().Y, L"Point Y value");
			Assert::AreEqual(expectedPointZ, hRec.GetPoint().Z, L"Point Z value");
			Assert::AreEqual(expectedNormalX, hRec.GetNormal().X, L"Normal X value");
			Assert::AreEqual(expectedNormalY, hRec.GetNormal().Y, L"Normal Y value");
			Assert::AreEqual(expectedNormalZ, hRec.GetNormal().Z, L"Normal Z value");

			delete sphere;
		}
	};
}