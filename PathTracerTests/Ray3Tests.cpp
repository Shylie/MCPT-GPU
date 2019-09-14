#include <CppUnitTest.h>
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

#include "..\\PathTracer\Common.h"

namespace PathTracerTests
{
	TEST_CLASS(Ray3Tests)
	{
	public:
		TEST_METHOD(Ray3Ctor)
		{
			const float oExpectedX = 2.0f, oExpectedY = 3.0f, oExpectedZ = 1.5f, dExpectedLength = 1.0f, dExpectedX = 0.9874838622020357632526696317373f, dExpectedY = 0.07053456158585982688037621165527f, dExpectedZ = 0.14106912317171965376075242331053f;

			Vec3 o(2.0f, 3.0f, 1.5f), d(2.8f, 0.2f, 0.4f);
			Ray3 ray(o, d);

			Assert::AreEqual(oExpectedX, ray.Origin().X, L"Origin X");
			Assert::AreEqual(oExpectedY, ray.Origin().Y, L"Origin Y");
			Assert::AreEqual(oExpectedZ, ray.Origin().Z, L"Origin Z");
			Assert::AreEqual(dExpectedLength, ray.Direction().Length(), L"Direction Length");
			Assert::AreEqual(dExpectedX, ray.Direction().X, L"Direction X");
			Assert::AreEqual(dExpectedY, ray.Direction().Y, L"Direction Y");
			Assert::AreEqual(dExpectedZ, ray.Direction().Z, L"Direction Z");
		}

		TEST_METHOD(Ray3PointAt)
		{
			const float expectedX = 1.0f + 2.0f * 0.80178372573727315405366044263926f, expectedY = 3.0f + 2.0f * 0.53452248382484876936910696175951f, expectedZ = -2.3f + 2.0f * 0.26726124191242438468455348087975f;

			float t = 2.0f;
			Vec3 o(1.0f, 3.0f, -2.3f), d(3.0f, 2.0f, 1.0f);
			Ray3 ray(o, d);

			Vec3 result = ray.PointAt(t);

			Assert::AreEqual(expectedX, result.X, 0.000001f, L"X");
			Assert::AreEqual(expectedY, result.Y, 0.000001f, L"Y");
			Assert::AreEqual(expectedZ, result.Z, 0.000001f, L"Z");
		}
	};
}