using System;
using System.Runtime.InteropServices;
using System.Linq;

namespace PathTracerNET
{
	internal sealed class PTObject
	{
		private PTObject(IntPtr ptr, PTObjectKind ptoType, params PTObject[] children)
		{
			this.ptr = ptr;
			Kind = ptoType;
			this.children = children;
		}

		public void Destroy(bool destroyChildren = true)
		{
			switch (Kind)
			{
				case PTObjectKind.Hittable:
					if (destroyChildren)
					{
						foreach (PTObject child in children)
						{
							child.Destroy();
						}
					}
					DestroyHittable(ptr);
					break;
				case PTObjectKind.Material:
					if (destroyChildren)
					{
						foreach (PTObject child in children)
						{
							child.Destroy();
						}
					}
					DestroyMaterial(ptr);
					break;
			}
			Kind = PTObjectKind.Invalid;
		}

		public PTObject Translate(Vec3 offset)
		{
			return Translation(offset, this);
		}

		public PTObject Rotate(float theta, Alignment alignment)
		{
			return Rotation(theta, alignment, this);
		}

		public PTObjectKind Kind { get; private set; } = PTObjectKind.Invalid;

		private readonly PTObject[] children;

		private readonly IntPtr ptr = IntPtr.Zero;

		public enum PTObjectKind
		{
			Invalid,
			Hittable,
			Material
		}

		[Flags]
		public enum Alignment
		{
			None = 0,
			X = 1 << 0,
			Y = 1 << 1,
			Z = 1 << 2
		}

		public static bool RenderScene(int width, int height, int samples, string fname, Vec3 lookFrom, Vec3 lookAt, Vec3 vup, float vfov, float aspect, PTObject scene)
		{
			if (scene.Kind != PTObjectKind.Hittable)
			{
				throw new ArgumentException("Invalid PTObjectKind: not a Hittable.", nameof(scene));
			}
			return RenderScene(width, height, samples, fname + ".ppm", lookFrom, lookAt, vup, vfov, aspect, scene.ptr);
		}

		public static bool RenderSceneChunked(int width, int height, int samples, int chunkSize, string fname, Vec3 lookFrom, Vec3 lookAt, Vec3 vup, float vfov, float aspect, PTObject scene)
		{
			if (scene.Kind != PTObjectKind.Hittable)
			{
				throw new ArgumentException("Invalid PTObjectKind: not a Hittable.", nameof(scene));
			}
			if (chunkSize % 4 != 0)
			{
				throw new ArgumentException("Invalid chunk size: not a multiple of 4.", nameof(chunkSize));
			}
			return RenderSceneChunked(width, height, samples, chunkSize, fname + ".ppm", lookFrom, lookAt, vup, vfov, aspect, scene.ptr);
		}

		public static PTObject HittableList(params PTObject[] ptObjects)
		{
			foreach (PTObject ptObject in ptObjects)
			{
				if (ptObject.Kind != PTObjectKind.Hittable) throw new ArgumentException("Invalid PTObjectKind: not a Hittable.", nameof(ptObject));
			}
			return new PTObject(ConstructHittableList(ptObjects.Length, ptObjects.Select(pto => pto.ptr).ToArray()), PTObjectKind.Hittable, ptObjects);
		}
		
		public static PTObject Translation(Vec3 offset, PTObject hittable)
		{
			if (hittable.Kind != PTObjectKind.Hittable) throw new ArgumentException("Invalid PTObjectKind: not a Hittable.", nameof(hittable));
			return new PTObject(ConstructTranslation(offset, hittable.ptr), PTObjectKind.Hittable, hittable);
		}

		public static PTObject Rotation(float theta, Alignment alignment, PTObject hittable)
		{
			if (hittable.Kind != PTObjectKind.Hittable) throw new ArgumentException("Invalid PTObjectKind: not a Hittable.", nameof(hittable));
			if (alignment != Alignment.X && alignment != Alignment.Y && alignment != Alignment.Z) throw new ArgumentException("Invalid PTObjectAlignment.", nameof(alignment));
			return new PTObject(ConstructRotation(theta, alignment, hittable.ptr), PTObjectKind.Hittable, hittable);
		}

		public static PTObject Sphere(Vec3 center, float radius, PTObject material)
		{
			if (material.Kind != PTObjectKind.Material) throw new ArgumentException("Invalid PTObjectKind: not a Material.", nameof(material));
			return new PTObject(ConstructSphere(center, radius, material.ptr), PTObjectKind.Hittable, material);
		}

		public static PTObject RectangularPrism(float x1, float x2, float y1, float y2, float z1, float z2, PTObject material)
		{
			if (material.Kind != PTObjectKind.Material) throw new ArgumentException("Invalid PTObjectKind: not a Material.", nameof(material));
			PTObject xyMin = RectangularPlane(x1, x2, y1, y2, z1, Alignment.X | Alignment.Y, false, true, material);
			PTObject xyMax = RectangularPlane(x1, x2, y1, y1, z2, Alignment.X | Alignment.Y, false, false, material);
			PTObject xzMin = RectangularPlane(x1, x2, z1, z2, y1, Alignment.X | Alignment.Z, false, true, material);
			PTObject xzMax = RectangularPlane(x1, x2, z1, z2, y2, Alignment.X | Alignment.Z, false, false, material);
			PTObject yzMin = RectangularPlane(y1, y2, z1, z2, x1, Alignment.Y | Alignment.Z, false, true, material);
			PTObject yzMax = RectangularPlane(y1, y2, z1, z2, x2, Alignment.Y | Alignment.Z, false, false, material);
			return HittableList(xyMin, xyMax, xzMin, xzMax, yzMin, yzMax);
		}

		public static PTObject RectangularPlane(float a1, float a2, float b1, float b2, float k, Alignment alignment, bool autoNormal, bool invertNormal, PTObject material)
		{
			if (material.Kind != PTObjectKind.Material) throw new ArgumentException("Invalid PTObjectKind: not a Material.", nameof(material));
			switch (alignment)
			{
				case Alignment.None:
				case Alignment.X:
				case Alignment.Y:
				case Alignment.Z:
				case Alignment.X | Alignment.Y | Alignment.Z:
					throw new ArgumentException("Invalid PTObjectAlignment.", nameof(alignment));
			}
			return new PTObject(ConstructRectangularPlane(a1, a2, b1, b2, k, alignment, autoNormal, invertNormal, material.ptr), PTObjectKind.Hittable, material);
		}

		public static PTObject TriangularPlane(float a1, float b1, float a2, float b2, float a3, float b3, float k, Alignment alignment, bool autoNormal, bool invertNormal, PTObject material)
		{
			if (material.Kind != PTObjectKind.Material) throw new ArgumentException("Invalid PTObjectKind: not a Material.", nameof(material));
			switch (alignment)
			{
				case Alignment.None:
				case Alignment.X:
				case Alignment.Y:
				case Alignment.Z:
				case Alignment.X | Alignment.Y | Alignment.Z:
					throw new ArgumentException("Invalid PTObjectAlignment.", nameof(alignment));
			}
			return new PTObject(ConstructTriangularPlane(a1, b1, a2, b2, a3, b3, k, alignment, autoNormal, invertNormal, material.ptr), PTObjectKind.Hittable, material);
		}

		public static PTObject DistortedSphere(Vec3 center, float radius, float frequency, float amplitude, PTObject material)
		{
			if (material.Kind != PTObjectKind.Material) throw new ArgumentException("Invalid PTObjectKind: not a Material.", nameof(material));
			return new PTObject(ConstructDistortedSphere(center, radius, frequency, amplitude, material.ptr), PTObjectKind.Hittable, material);
		}

		public static PTObject Lambertian(float r, float g, float b)
		{
			return new PTObject(ConstructLambertian(r, g, b), PTObjectKind.Material);
		}

		public static PTObject Metal(float r, float g, float b, float fuzz)
		{
			return new PTObject(ConstructMetal(r, g, b, fuzz), PTObjectKind.Material);
		}

		public static PTObject Dieletric(float r, float g, float b, float refractiveIndex)
		{
			return new PTObject(ConstructDielectric(r, g, b, refractiveIndex), PTObjectKind.Material);
		}

		public static PTObject DiffuseLight(float r, float g, float b)
		{
			return new PTObject(ConstructDiffuseLight(r, g, b), PTObjectKind.Material);
		}

		[DllImport("PathTracer.dll")]
		private static extern bool RenderScene(int width, int height, int samples, string fname, Vec3 lookFrom, Vec3 lookAt, Vec3 vup, float vfov, float aspect, IntPtr scene);

		[DllImport("PathTracer.dll")]
		private static extern bool RenderSceneChunked(int width, int height, int samples, int chunkSize, string fname, Vec3 lookFrom, Vec3 lookAt, Vec3 vup, float vfov, float aspect, IntPtr scene);

		[DllImport("PathTracer.dll")]
		private static extern IntPtr ConstructHittableList(int numHittables, [MarshalAs(UnmanagedType.LPArray)] IntPtr[] hittables);

		[DllImport("PathTracer.dll")]
		private static extern IntPtr ConstructTranslation(Vec3 offset, IntPtr hittable);

		[DllImport("PathTracer.dll")]
		private static extern IntPtr ConstructRotation(float theta, Alignment alignment, IntPtr hittable);

		[DllImport("PathTracer.dll")]
		private static extern IntPtr ConstructSphere(Vec3 center, float radius, IntPtr mat);

		[DllImport("PathTracer.dll")]
		private static extern IntPtr ConstructRectangularPlane(float a1, float a2, float b1, float b2, float k, Alignment alignment, bool autoNormal, bool invertNormal, IntPtr mat);

		[DllImport("PathTracer.dll")]
		private static extern IntPtr ConstructTriangularPlane(float a1, float b1, float a2, float b2, float a3, float b3, float k, Alignment alignment, bool autoNormal, bool invertNormal, IntPtr mat);

		[DllImport("PathTracer.dll")]
		private static extern IntPtr ConstructDistortedSphere(Vec3 center, float radius, float frequency, float amplitude, IntPtr mat);

		[DllImport("PathTracer.dll")]
		private static extern IntPtr ConstructLambertian(float r, float g, float b);

		[DllImport("PathTracer.dll")]
		private static extern IntPtr ConstructMetal(float r, float g, float b, float fuzz);

		[DllImport("PathTracer.dll")]
		private static extern IntPtr ConstructDielectric(float r, float g, float b, float refractiveIndex);

		[DllImport("PathTracer.dll")]
		private static extern IntPtr ConstructDiffuseLight(float r, float g, float b);

		[DllImport("PathTracer.dll")]
		private static extern void DestroyHittable(IntPtr ptr);

		[DllImport("PathTracer.dll")]
		private static extern void DestroyMaterial(IntPtr ptr);
	}
}
