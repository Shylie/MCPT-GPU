using System;
using System.Runtime.InteropServices;

namespace PathTracerNET
{
	[StructLayout(LayoutKind.Sequential)]
	internal struct Vec3
	{
		public Vec3(float val)
		{
			X = val;
			Y = val;
			Z = val;
		}

		public Vec3(float x, float y, float z)
		{
			X = x;
			Y = y;
			Z = z;
		}

		public float X { get; set; }
		public float Y { get; set; }
		public float Z { get; set; }

		public float Length => V3Length(this);
		public float LengthSquared => V3LengthSquared(this);

		public Vec3 Normalized => V3Normalized(this);

		public static float Dot(Vec3 va, Vec3 vb) => V3Dot(va, vb);
		public static Vec3 Cross(Vec3 va, Vec3 vb) => V3Cross(va, vb);

		public static Vec3 operator -(Vec3 vec) => V3OpNegate(vec);
		public static Vec3 operator +(Vec3 va, Vec3 vb) => V3OpAdd(va, vb);
		public static Vec3 operator -(Vec3 va, Vec3 vb) => V3OpSub(va, vb);
		public static Vec3 operator *(Vec3 va, Vec3 vb) => V3OpMul(va, vb);
		public static Vec3 operator /(Vec3 va, Vec3 vb) => V3OpDiv(va, vb);
		public static Vec3 operator *(Vec3 vec, float scalar) => V3OpScalarMul(vec, scalar);
		public static Vec3 operator *(float scalar, Vec3 vec) => V3OpScalarMul(vec, scalar);
		public static Vec3 operator /(Vec3 vec, float scalar) => V3OpScalarDiv(vec, scalar);
		public static Vec3 operator /(float scalar, Vec3 vec) => V3OpScalarDiv(vec, scalar);

		[DllImport("PathTracer.dll")]
		private static extern float V3Length(Vec3 vec);

		[DllImport("PathTracer.dll")]
		private static extern float V3LengthSquared(Vec3 vec);

		[DllImport("PathTracer.dll")]
		private static extern Vec3 V3Normalized(Vec3 vec);

		[DllImport("PathTracer.dll")]
		private static extern float V3Dot(Vec3 va, Vec3 vb);

		[DllImport("PathTracer.dll")]
		private static extern Vec3 V3Cross(Vec3 va, Vec3 vb);

		[DllImport("PathTracer.dll")]
		private static extern Vec3 V3OpNegate(Vec3 vec);

		[DllImport("PathTracer.dll")]
		private static extern Vec3 V3OpAdd(Vec3 va, Vec3 vb);

		[DllImport("PathTracer.dll")]
		private static extern Vec3 V3OpSub(Vec3 va, Vec3 vb);

		[DllImport("PathTracer.dll")]
		private static extern Vec3 V3OpMul(Vec3 va, Vec3 vb);

		[DllImport("PathTracer.dll")]
		private static extern Vec3 V3OpDiv(Vec3 va, Vec3 vb);

		[DllImport("PathTracer.dll")]
		private static extern Vec3 V3OpScalarMul(Vec3 vec, float scalar);

		[DllImport("PathTracer.dll")]
		private static extern Vec3 V3OpScalarDiv(Vec3 vec, float scalar);
	}
}
