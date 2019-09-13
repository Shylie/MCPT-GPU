#include <CppUnitTest.h>
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

#include "Common.h"

namespace Tests
{
	TEST_CLASS(Vec3Tests)
	{
	public:
		TEST_METHOD(Vec3DefaultCtor)
		{
			const float expected = 0.0f;

			Vec3 v3;

			Assert::AreEqual(expected, v3.X);
			Assert::AreEqual(expected, v3.Y);
			Assert::AreEqual(expected, v3.Z);
		}

		TEST_METHOD(Vec3SingleCompCtor)
		{
			const float expected = 1.0f;

			Vec3 v3(1.0f);

			Assert::AreEqual(expected, v3.X);
			Assert::AreEqual(expected, v3.Y);
			Assert::AreEqual(expected, v3.Z);
		}

		TEST_METHOD(Vec3AllCompCtor)
		{
			const float expectedX = 1.0f, expectedY = 2.0f, expectedZ = 3.0f;

			Vec3 v3(1.0f, 2.0f, 3.0f);

			Assert::AreEqual(expectedX, v3.X);
			Assert::AreEqual(expectedY, v3.Y);
			Assert::AreEqual(expectedZ, v3.Z);
		}

		TEST_METHOD(Vec3Length)
		{
			const float expected = 5.3851648071345040312507104915403f;
			
			Vec3 v3(2.0f, 3.0f, 4.0f);

			Assert::AreEqual(expected, v3.Length());
		}

		TEST_METHOD(Vec3LengthSquared)
		{
			const float expected = 29.0f;

			Vec3 v3(2.0f, 3.0f, 4.0f);

			Assert::AreEqual(expected, v3.LengthSquared());
		}

		TEST_METHOD(Vec3Normalized)
		{
			const float expected = 0.57735026918962576450914878050196f;

			Vec3 v3(2.0f, 2.0f, 2.0f);
			Vec3 v3n = v3.Normalized();

			Assert::AreEqual(expected, v3n.X);
			Assert::AreEqual(expected, v3n.Y);
			Assert::AreEqual(expected, v3n.Z);
		}

		TEST_METHOD(Vec3Dot)
		{
			const float expected = 98.0f;

			Vec3 v1(2.0f, 4.0f, 8.0f), v2(3.0f, 5.0f, 9.0f);

			Assert::AreEqual(expected, Vec3::Dot(v1, v2));
		}

		TEST_METHOD(Vec3Cross)
		{
			const float expectedX = -4.0f, expectedY = 6.0f, expectedZ = -2.0f;

			Vec3 v1(2.0f, 4.0f, 8.0f), v2(3.0f, 5.0f, 9.0f);
			Vec3 cp = Vec3::Cross(v1, v2);

			Assert::AreEqual(expectedX, cp.X);
			Assert::AreEqual(expectedY, cp.Y);
			Assert::AreEqual(expectedZ, cp.Z);
		}

		TEST_METHOD(Vec3Negate)
		{
			const float expectedX = 2.0f, expectedY = -3.0f, expectedZ = 5.0f;

			Vec3 v3(-2.0f, 3.0f, -5.0f);

			Assert::AreEqual(expectedX, (-v3).X);
			Assert::AreEqual(expectedY, (-v3).Y);
			Assert::AreEqual(expectedZ, (-v3).Z);
		}

		TEST_METHOD(Vec3AddVec3)
		{
			const float expectedX = 2.0f, expectedY = 4.0f, expectedZ = 5.0f;

			Vec3 v1(1.0f, 3.0f, 2.0f), v2(1.0f, 1.0f, 3.0f);
			Vec3 result = v1 + v2;

			Assert::AreEqual(expectedX, result.X);
			Assert::AreEqual(expectedY, result.Y);
			Assert::AreEqual(expectedZ, result.Z);
		}

		TEST_METHOD(Vec3SubVec3)
		{
			const float expectedX = 2.0f, expectedY = 4.0f, expectedZ = 5.0f;

			Vec3 v1(3.0f, 5.0f, 8.0f), v2(1.0f, 1.0f, 3.0f);
			Vec3 result = v1 - v2;

			Assert::AreEqual(expectedX, result.X);
			Assert::AreEqual(expectedY, result.Y);
			Assert::AreEqual(expectedZ, result.Z);
		}

		TEST_METHOD(Vec3MulVec3)
		{
			const float expectedX = 2.0f, expectedY = 9.0f, expectedZ = 4.0f;

			Vec3 v1(1.0f, 3.0f, 2.0f), v2(2.0f, 3.0f, 2.0f);
			Vec3 result = v1 * v2;

			Assert::AreEqual(expectedX, result.X);
			Assert::AreEqual(expectedY, result.Y);
			Assert::AreEqual(expectedZ, result.Z);
		}

		TEST_METHOD(Vec3DivVec3)
		{
			const float expectedX = 1.0f, expectedY = 1.5f, expectedZ = 3.0f;

			Vec3 v1(1.0f, 3.0f, 9.0f), v2(1.0f, 2.0f, 3.0f);
			Vec3 result = v1 / v2;

			Assert::AreEqual(expectedX, result.X);
			Assert::AreEqual(expectedY, result.Y);
			Assert::AreEqual(expectedZ, result.Z);
		}

		TEST_METHOD(Vec3MulFloat)
		{
			const float expectedX = 4.0f, expectedY = 6.0f, expectedZ = 12.0f;

			Vec3 v1(2.0f, 3.0f, 6.0f);
			float scalar = 2.0f;

			Vec3 result = v1 * scalar;

			Assert::AreEqual(expectedX, result.X);
			Assert::AreEqual(expectedY, result.Y);
			Assert::AreEqual(expectedZ, result.Z);
		}

		TEST_METHOD(Vec3DivFloat)
		{
			const float expectedX = 1.0f, expectedY = 1.5f, expectedZ = 3.0f;

			Vec3 v1(2.0f, 3.0f, 6.0f);
			float scalar = 2.0f;

			Vec3 result = v1 / scalar;

			Assert::AreEqual(expectedX, result.X);
			Assert::AreEqual(expectedY, result.Y);
			Assert::AreEqual(expectedZ, result.Z);
		}
	};
}