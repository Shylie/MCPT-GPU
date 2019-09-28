#include <CppUnitTest.h>
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

#include "..\\PathTracer\Common.h"
#include "..\\PathTracer\Sphere.h"
#include "..\\PathTracer\Translation.h"

namespace PathTracerTests
{
	TEST_CLASS(TranslationTests)
	{
	public:
		TEST_METHOD(TranslationCtor)
		{
			Hittable* sphere = new Sphere(Vec3(0.0f), 1.0f, (Material*)nullptr);
			Hittable* translated = new Translation(Vec3(1.0f), sphere);

			Assert::IsNotNull(translated->GetPtrGPU());

			delete sphere;
			delete translated;
		}

		TEST_METHOD(TranslationSphereHitFromOutside)
		{
			const float expectedT = 4.1961524227066318805823390245176f;
			const float expectedPointX = 3.0f - 0.57735026918962576450914878050196f;
			const float expectedPointY = 3.0f - 0.57735026918962576450914878050196f;
			const float expectedPointZ = 3.0f - 0.57735026918962576450914878050196f;
			const float expectedNormalX = -0.57735026918962576450914878050196f;
			const float expectedNormalY = -0.57735026918962576450914878050196f;
			const float expectedNormalZ = -0.57735026918962576450914878050196f;

			Hittable* sphere = new Sphere(Vec3(2.0f), 1.0f, (Material*)nullptr);
			Hittable* translated = new Translation(Vec3(1.0f), sphere);

			HitRecord hRec;

			Ray3 ray(Vec3(0.0f), Vec3(1.0f));

			bool hit = translated->Hit(ray, 0.001f, FLT_MAX, hRec);

			Assert::IsTrue(hit, L"Hit sphere");
			Assert::AreEqual(expectedT, hRec.GetT(), 0.00001f, L"T value");
			Assert::AreEqual(expectedPointX, hRec.GetPoint().X, 0.00001f, L"Point X value");
			Assert::AreEqual(expectedPointY, hRec.GetPoint().Y, 0.00001f, L"Point Y value");
			Assert::AreEqual(expectedPointZ, hRec.GetPoint().Z, 0.00001f, L"Point Z value");
			Assert::AreEqual(expectedNormalX, hRec.GetNormal().X, 0.00001f, L"Normal X value");
			Assert::AreEqual(expectedNormalY, hRec.GetNormal().Y, 0.00001f, L"Normal Y value");
			Assert::AreEqual(expectedNormalZ, hRec.GetNormal().Z, 0.00001f, L"Normal Z value");

			delete sphere;
			delete translated;
		}

		TEST_METHOD(TranslationSphereMissFromOutside)
		{
			Hittable* sphere = new Sphere(Vec3(2.0f), 1.0f, (Material*)nullptr);
			Hittable* translation = new Translation(Vec3(-1.0f), sphere);

			HitRecord hRec;

			Ray3 ray(Vec3(0.0f), Vec3(-1.0f));

			bool hit = translation->Hit(ray, 0.001f, FLT_MAX, hRec);

			Assert::IsFalse(hit, L"Hit sphere");

			delete sphere;
			delete translation;
		}

		TEST_METHOD(TranslationSphereHitFromInside)
		{
			const float expectedT = 1.8660254037844386467637231707529f;
			const float expectedPointX = 2.5f + 0.57735026918962576450914878050196f;
			const float expectedPointY = 2.5f + 0.57735026918962576450914878050196f;
			const float expectedPointZ = 2.5f + 0.57735026918962576450914878050196f;
			const float expectedNormalX = 0.57735026918962576450914878050196f;
			const float expectedNormalY = 0.57735026918962576450914878050196f;
			const float expectedNormalZ = 0.57735026918962576450914878050196f;

			Hittable* sphere = new Sphere(Vec3(2.0f), 1.0f, (Material*)nullptr);
			Hittable* translated = new Translation(Vec3(0.5f), sphere);

			HitRecord hRec;

			Ray3 ray(Vec3(2.0f), Vec3(1.0f));

			bool hit = translated->Hit(ray, 0.001f, FLT_MAX, hRec);

			Assert::IsTrue(hit, L"Hit sphere");
			Assert::AreEqual(expectedT, hRec.GetT(), L"T value");
			Assert::AreEqual(expectedPointX, hRec.GetPoint().X, 0.00001f,  L"Point X value");
			Assert::AreEqual(expectedPointY, hRec.GetPoint().Y, 0.00001f, L"Point Y value");
			Assert::AreEqual(expectedPointZ, hRec.GetPoint().Z, 0.00001f, L"Point Z value");
			Assert::AreEqual(expectedNormalX, hRec.GetNormal().X, 0.00001f, L"Normal X value");
			Assert::AreEqual(expectedNormalY, hRec.GetNormal().Y, 0.00001f, L"Normal Y value");
			Assert::AreEqual(expectedNormalZ, hRec.GetNormal().Z, 0.00001f, L"Normal Z value");

			delete sphere;
			delete translated;
		}
	};
}