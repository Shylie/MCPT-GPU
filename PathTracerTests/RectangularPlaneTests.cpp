#include <CppUnitTest.h>
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

#include "..\\PathTracer\Common.h"
#include "..\\PathTracer\RectangularPlane.h"

namespace PathTracerTests
{
	TEST_CLASS(RectangularPlaneTests)
	{
	public:
		TEST_METHOD(RectangularPlaneCtor)
		{
			Hittable* rectangularPlane = new RectangularPlane(0.0f, 1.0f, 0.0f, 1.0f, 0.0f, (Alignment)(X | Y), true, false, nullptr);

			Assert::IsNotNull(rectangularPlane->GetPtrGPU());

			delete rectangularPlane;
		}

		TEST_METHOD(RectangularPlaneHitAutoNormal)
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

			Hittable* rectangularPlane = new RectangularPlane(-1.0f, 1.0f, -1.0f, 1.0f, 0.0f, (Alignment)(X | Z), true, false, nullptr);

			Vec3 originTop(0.0f, 1.0f, 0.0f);
			Vec3 directionTop(0.0f, -1.0f, 0.0f);
			Vec3 originBottom(0.0f, -1.0f, 0.0f);
			Vec3 directionBottom(0.0f, 1.0f, 0.0f);
			
			Ray3 top(originTop, directionTop);
			Ray3 bottom(originBottom, directionBottom);

			HitRecord hRecTop;
			HitRecord hRecBottom;

			bool hitTop = rectangularPlane->Hit(top, 0.001f, FLT_MAX, hRecTop);
			bool hitBottom = rectangularPlane->Hit(bottom, 0.001f, FLT_MAX, hRecBottom);

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

			delete rectangularPlane;
		}

		TEST_METHOD(RectangularPlaneHitNormal)
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

			Hittable* rectangularPlane = new RectangularPlane(-1.0f, 1.0f, -1.0f, 1.0f, 0.0f, (Alignment)(X | Z), false, false, nullptr);

			Vec3 originTop(0.0f, 1.0f, 0.0f);
			Vec3 directionTop(0.0f, -1.0f, 0.0f);
			Vec3 originBottom(0.0f, -1.0f, 0.0f);
			Vec3 directionBottom(0.0f, 1.0f, 0.0f);

			Ray3 top(originTop, directionTop);
			Ray3 bottom(originBottom, directionBottom);

			HitRecord hRecTop;
			HitRecord hRecBottom;

			bool hitTop = rectangularPlane->Hit(top, 0.001f, FLT_MAX, hRecTop);
			bool hitBottom = rectangularPlane->Hit(bottom, 0.001f, FLT_MAX, hRecBottom);

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

			delete rectangularPlane;
		}

		TEST_METHOD(RectangularPlaneHitInvertNormal)
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

			Hittable* rectangularPlane = new RectangularPlane(-1.0f, 1.0f, -1.0f, 1.0f, 0.0f, (Alignment)(X | Z), false, true, nullptr);

			Vec3 originTop(0.0f, 1.0f, 0.0f);
			Vec3 directionTop(0.0f, -1.0f, 0.0f);
			Vec3 originBottom(0.0f, -1.0f, 0.0f);
			Vec3 directionBottom(0.0f, 1.0f, 0.0f);

			Ray3 top(originTop, directionTop);
			Ray3 bottom(originBottom, directionBottom);

			HitRecord hRecTop;
			HitRecord hRecBottom;

			bool hitTop = rectangularPlane->Hit(top, 0.001f, FLT_MAX, hRecTop);
			bool hitBottom = rectangularPlane->Hit(bottom, 0.001f, FLT_MAX, hRecBottom);

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

			delete rectangularPlane;
		}

		TEST_METHOD(RectangularPlaneMiss)
		{
			Hittable* rectangularPlane = new RectangularPlane(2.0f, 4.0f, 2.0f, 4.0f, 2.0f, (Alignment)(X | Z), true, false, nullptr);

			Vec3 origin(0.0f);
			Vec3 direction(0.0f, 1.0f, 0.0f);
			Ray3 ray(origin, direction);

			HitRecord hRec;

			bool hit = rectangularPlane->Hit(ray, 0.001f, FLT_MAX, hRec);

			Assert::IsFalse(hit, L"Hit");

			delete rectangularPlane;
		}
	};
}