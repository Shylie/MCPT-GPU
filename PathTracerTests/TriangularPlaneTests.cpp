#include <CppUnitTest.h>
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

#include "..\\PathTracer\Common.h"
#include "..\\PathTracer\TriangularPlane.h"

namespace PathTracerTests
{
	TEST_CLASS(TriangularPlaneTests)
	{
	public:
		TEST_METHOD(TriangularPlaneCtor)
		{
			Hittable* triangularPlane = new TriangularPlane(0.0f, 1.0f, 1.0f, 2.0f, 2.0f, 1.0f, 0.0f, (Alignment)(X | Y), true, false, (Material*)nullptr);

			Assert::IsNotNull(triangularPlane->GetPtrGPU());

			delete triangularPlane;
		}

		TEST_METHOD(TriangularPlaneHitAutoNormal)
		{
			const float expectedT = 1.0f;
			const float expectedX = 0.0f;
			const float expectedY = 0.0f;
			const float expectedZ = 0.0f;
			const float expectedTopNormalX = 0.0f;
			const float expectedTopNormalY = 1.0f;
			const float expectedTopNormalZ = 0.0f;
			const float expectedBottomNormalX = 0.0f;
			const float expectedBottomNormalY = -1.0f;
			const float expectedBottomNormalZ = 0.0f;

			Hittable* triangularPlane = new TriangularPlane(-2.0f, -2.0f, 0.0f, 2.0f, 2.0f, -2.0f, 0.0f, (Alignment)(X | Z), true, false, (Material*)nullptr);

			Vec3 originTop(0.0f, 1.0f, 0.0f);
			Vec3 directionTop(0.0f, -1.0f, 0.0f);
			Vec3 originBottom(0.0f, -1.0f, 0.0f);
			Vec3 directionBottom(0.0f, 1.0f, 0.0f);

			Ray3 top(originTop, directionTop);
			Ray3 bottom(originBottom, directionBottom);

			HitRecord hRecTop;
			HitRecord hRecBottom;

			bool hitTop = triangularPlane->Hit(top, 0.001f, FLT_MAX, hRecTop);
			bool hitBottom = triangularPlane->Hit(bottom, 0.001f, FLT_MAX, hRecBottom);

			Assert::IsTrue(hitTop, L"Hit from top");
			Assert::IsTrue(hitBottom, L"Hit from bottom");

			Assert::AreEqual(expectedT, hRecTop.GetT(), L"Top T value");
			Assert::AreEqual(expectedT, hRecBottom.GetT(), L"Bottom T value");

			Assert::AreEqual(expectedX, hRecTop.GetPoint().X, L"Top X value");
			Assert::AreEqual(expectedY, hRecTop.GetPoint().Y, L"Top Y value");
			Assert::AreEqual(expectedZ, hRecTop.GetPoint().Z, L"Top Z value");
			Assert::AreEqual(expectedX, hRecBottom.GetPoint().X, L"Bottom X value");
			Assert::AreEqual(expectedY, hRecBottom.GetPoint().Y, L"Bottom Y value");
			Assert::AreEqual(expectedZ, hRecBottom.GetPoint().Z, L"Bottom Z value");

			Assert::AreEqual(expectedTopNormalX, hRecTop.GetNormal().X, L"Top normal X value");
			Assert::AreEqual(expectedTopNormalY, hRecTop.GetNormal().Y, L"Top normal Y value");
			Assert::AreEqual(expectedTopNormalZ, hRecTop.GetNormal().Z, L"Top normal Z value");
			Assert::AreEqual(expectedBottomNormalX, hRecBottom.GetNormal().X, L"Bottom normal X value");
			Assert::AreEqual(expectedBottomNormalY, hRecBottom.GetNormal().Y, L"Bottom normal Y value");
			Assert::AreEqual(expectedBottomNormalZ, hRecBottom.GetNormal().Z, L"Bottom normal Z value");

			delete triangularPlane;
		}

		TEST_METHOD(TriangularPlaneHitNormal)
		{
			const float expectedT = 1.0f;
			const float expectedX = 0.0f;
			const float expectedY = 0.0f;
			const float expectedZ = 0.0f;
			const float expectedTopNormalX = 0.0f;
			const float expectedTopNormalY = 1.0f;
			const float expectedTopNormalZ = 0.0f;
			const float expectedBottomNormalX = 0.0f;
			const float expectedBottomNormalY = 1.0f;
			const float expectedBottomNormalZ = 0.0f;

			Hittable* triangularPlane = new TriangularPlane(-2.0f, -2.0f, 0.0f, 2.0f, 2.0f, -2.0f, 0.0f, (Alignment)(X | Z), false, false, (Material*)nullptr);

			Vec3 originTop(0.0f, 1.0f, 0.0f);
			Vec3 directionTop(0.0f, -1.0f, 0.0f);
			Vec3 originBottom(0.0f, -1.0f, 0.0f);
			Vec3 directionBottom(0.0f, 1.0f, 0.0f);

			Ray3 top(originTop, directionTop);
			Ray3 bottom(originBottom, directionBottom);

			HitRecord hRecTop;
			HitRecord hRecBottom;

			bool hitTop = triangularPlane->Hit(top, 0.001f, FLT_MAX, hRecTop);
			bool hitBottom = triangularPlane->Hit(bottom, 0.001f, FLT_MAX, hRecBottom);

			Assert::IsTrue(hitTop, L"Hit from top");
			Assert::IsTrue(hitBottom, L"Hit from bottom");

			Assert::AreEqual(expectedT, hRecTop.GetT(), L"Top T value");
			Assert::AreEqual(expectedT, hRecBottom.GetT(), L"Bottom T value");

			Assert::AreEqual(expectedX, hRecTop.GetPoint().X, L"Top X value");
			Assert::AreEqual(expectedY, hRecTop.GetPoint().Y, L"Top Y value");
			Assert::AreEqual(expectedZ, hRecTop.GetPoint().Z, L"Top Z value");
			Assert::AreEqual(expectedX, hRecBottom.GetPoint().X, L"Bottom X value");
			Assert::AreEqual(expectedY, hRecBottom.GetPoint().Y, L"Bottom Y value");
			Assert::AreEqual(expectedZ, hRecBottom.GetPoint().Z, L"Bottom Z value");

			Assert::AreEqual(expectedTopNormalX, hRecTop.GetNormal().X, L"Top normal X value");
			Assert::AreEqual(expectedTopNormalY, hRecTop.GetNormal().Y, L"Top normal Y value");
			Assert::AreEqual(expectedTopNormalZ, hRecTop.GetNormal().Z, L"Top normal Z value");
			Assert::AreEqual(expectedBottomNormalX, hRecBottom.GetNormal().X, L"Bottom normal X value");
			Assert::AreEqual(expectedBottomNormalY, hRecBottom.GetNormal().Y, L"Bottom normal Y value");
			Assert::AreEqual(expectedBottomNormalZ, hRecBottom.GetNormal().Z, L"Bottom normal Z value");

			delete triangularPlane;
		}

		TEST_METHOD(TriangularPlaneHitInvertNormal)
		{
			const float expectedT = 1.0f;
			const float expectedX = 0.0f;
			const float expectedY = 0.0f;
			const float expectedZ = 0.0f;
			const float expectedTopNormalX = 0.0f;
			const float expectedTopNormalY = -1.0f;
			const float expectedTopNormalZ = 0.0f;
			const float expectedBottomNormalX = 0.0f;
			const float expectedBottomNormalY = -1.0f;
			const float expectedBottomNormalZ = 0.0f;

			Hittable* triangularPlane = new TriangularPlane(-2.0f, -2.0f, 0.0f, 2.0f, 2.0f, -2.0f, 0.0f, (Alignment)(X | Z), false, true, (Material*)nullptr);

			Vec3 originTop(0.0f, 1.0f, 0.0f);
			Vec3 directionTop(0.0f, -1.0f, 0.0f);
			Vec3 originBottom(0.0f, -1.0f, 0.0f);
			Vec3 directionBottom(0.0f, 1.0f, 0.0f);

			Ray3 top(originTop, directionTop);
			Ray3 bottom(originBottom, directionBottom);

			HitRecord hRecTop;
			HitRecord hRecBottom;

			bool hitTop = triangularPlane->Hit(top, 0.001f, FLT_MAX, hRecTop);
			bool hitBottom = triangularPlane->Hit(bottom, 0.001f, FLT_MAX, hRecBottom);

			Assert::IsTrue(hitTop, L"Hit from top");
			Assert::IsTrue(hitBottom, L"Hit from bottom");

			Assert::AreEqual(expectedT, hRecTop.GetT(), L"Top T value");
			Assert::AreEqual(expectedT, hRecBottom.GetT(), L"Bottom T value");

			Assert::AreEqual(expectedX, hRecTop.GetPoint().X, L"Top X value");
			Assert::AreEqual(expectedY, hRecTop.GetPoint().Y, L"Top Y value");
			Assert::AreEqual(expectedZ, hRecTop.GetPoint().Z, L"Top Z value");
			Assert::AreEqual(expectedX, hRecBottom.GetPoint().X, L"Bottom X value");
			Assert::AreEqual(expectedY, hRecBottom.GetPoint().Y, L"Bottom Y value");
			Assert::AreEqual(expectedZ, hRecBottom.GetPoint().Z, L"Bottom Z value");

			Assert::AreEqual(expectedTopNormalX, hRecTop.GetNormal().X, L"Top normal X value");
			Assert::AreEqual(expectedTopNormalY, hRecTop.GetNormal().Y, L"Top normal Y value");
			Assert::AreEqual(expectedTopNormalZ, hRecTop.GetNormal().Z, L"Top normal Z value");
			Assert::AreEqual(expectedBottomNormalX, hRecBottom.GetNormal().X, L"Bottom normal X value");
			Assert::AreEqual(expectedBottomNormalY, hRecBottom.GetNormal().Y, L"Bottom normal Y value");
			Assert::AreEqual(expectedBottomNormalZ, hRecBottom.GetNormal().Z, L"Bottom normal Z value");

			delete triangularPlane;
		}

		TEST_METHOD(TriangularPlaneMiss)
		{
			Hittable* triangularPlane = new TriangularPlane(2.0f, 2.0f, 4.0f, 4.0f, 6.0f, 2.0f, 2.0f, (Alignment)(X | Z), true, false, (Material*)nullptr);

			Vec3 origin(2.5f, 0.0f, 3.5f);
			Vec3 direction(0.0f, 1.0f, 0.0f);
			Ray3 ray(origin, direction);

			HitRecord hRec;

			bool hit = triangularPlane->Hit(ray, 0.001f, FLT_MAX, hRec);

			Assert::IsFalse(hit, L"Hit");

			delete triangularPlane;
		}
	};
}