using System;
using System.Runtime.InteropServices;
using System.Xml.Serialization;

using PathTracerNET.Hittables;
using PathTracerNET.Hittables.Geometric;
using PathTracerNET.Hittables.Modifier;
using PathTracerNET.Hittables.SDF;
using PathTracerNET.Hittables.Plane;
using PathTracerNET.Materials;

namespace PathTracerNET
{
	[Serializable]
	[XmlInclude(typeof(Dieletric))]
	[XmlInclude(typeof(DiffuseLight))]
	[XmlInclude(typeof(Lambertian))]
	[XmlInclude(typeof(Metal))]
	[XmlInclude(typeof(Sphere))]
	[XmlInclude(typeof(Rotation))]
	[XmlInclude(typeof(Translation))]
	[XmlInclude(typeof(RectangularPlane))]
	[XmlInclude(typeof(TriangularPlane))]
	[XmlInclude(typeof(DistortedSphere))]
	[XmlInclude(typeof(HittableList))]
	public abstract class PTObject
	{
		#region PROPERTIES
		[XmlIgnore]
		public bool Valid
		{
			get
			{
				return _valid;
			}
			set
			{
				if (_valid && !value) Invalidated?.Invoke(this);
				_valid = value;
			}
		}

		[XmlIgnore]
		public abstract PTObjectKind Kind { get; }

		[XmlIgnore]
		protected internal IntPtr Pointer
		{
			get
			{
				if (!Valid)
				{
					_pointer = Init();
					Valid = true;
				}
				return _pointer;
			}
			set
			{
				_pointer = value;
			}
		}
		#endregion

		#region FIELDS
		[XmlIgnore]
		private IntPtr _pointer = IntPtr.Zero;

		[XmlIgnore]
		private bool _valid = false;
		#endregion

		#region EVENTS
		internal delegate void InvalidatedEvent(PTObject sender);

		internal event InvalidatedEvent Invalidated;
		#endregion

		#region METHODS
		internal abstract IntPtr Init();

		public void Destroy()
		{
			switch (Kind)
			{
				case PTObjectKind.Hittable:
					DestroyHittable(Pointer);
					break;
				case PTObjectKind.Material:
					DestroyMaterial(Pointer);
					break;
			}
			Valid = false;
		}

		public PTObject Translate(Vec3 offset)
		{
			return new Translation(offset, this);
		}

		public PTObject Rotate(float theta, Alignment alignment)
		{
			return new Rotation(theta, alignment, this);
		}
		#endregion

		#region ENUMS
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
		#endregion

		#region STATIC METHODS
		public static bool RenderScene(int width, int height, int samples, string fname, Vec3 lookFrom, Vec3 lookAt, Vec3 vup, float vfov, float aspect, float aperture, float lensRadius, PTObject scene)
		{
			if (scene.Kind != PTObjectKind.Hittable)
			{
				throw new ArgumentException("Invalid PTObjectKind: not a Hittable.", nameof(scene));
			}
			return RenderScene(width, height, samples, fname + ".ppm", lookFrom, lookAt, vup, vfov, aspect, aperture, lensRadius, scene.Pointer);
		}

		public static bool RenderSceneChunked(int width, int height, int samples, int chunkSize, string fname, Vec3 lookFrom, Vec3 lookAt, Vec3 vup, float vfov, float aspect, float aperture, float lensRadius, PTObject scene)
		{
			if (scene.Kind != PTObjectKind.Hittable)
			{
				throw new ArgumentException("Invalid PTObjectKind: not a Hittable.", nameof(scene));
			}
			if (chunkSize % 4 != 0)
			{
				throw new ArgumentException("Invalid chunk size: not a multiple of 4.", nameof(chunkSize));
			}
			return RenderSceneChunked(width, height, samples, chunkSize, fname + ".ppm", lookFrom, lookAt, vup, vfov, aspect, aperture, lensRadius, scene.Pointer);
		}
		#endregion

		#region EXTERNAL METHODS
		[DllImport("PathTracer.dll")]
		private static extern bool RenderScene(int width, int height, int samples, string fname, Vec3 lookFrom, Vec3 lookAt, Vec3 vup, float vfov, float aspect, float aperture, float lensRadius, IntPtr scene);

		[DllImport("PathTracer.dll")]
		private static extern bool RenderSceneChunked(int width, int height, int samples, int chunkSize, string fname, Vec3 lookFrom, Vec3 lookAt, Vec3 vup, float vfov, float aspect, float aperture, float lensRadius, IntPtr scene);

		[DllImport("PathTracer.dll")]
		protected static extern IntPtr ConstructHittableList(int numHittables, [MarshalAs(UnmanagedType.LPArray)] IntPtr[] hittables);

		[DllImport("PathTracer.dll")]
		protected static extern IntPtr ConstructTranslation(Vec3 offset, IntPtr hittable);

		[DllImport("PathTracer.dll")]
		protected static extern IntPtr ConstructRotation(float theta, Alignment alignment, IntPtr hittable);

		[DllImport("PathTracer.dll")]
		protected static extern IntPtr ConstructSphere(Vec3 center, float radius, IntPtr mat);

		[DllImport("PathTracer.dll")]
		protected static extern IntPtr ConstructRectangularPlane(float a1, float a2, float b1, float b2, float k, Alignment alignment, bool autoNormal, bool invertNormal, IntPtr mat);

		[DllImport("PathTracer.dll")]
		protected static extern IntPtr ConstructTriangularPlane(float a1, float b1, float a2, float b2, float a3, float b3, float k, Alignment alignment, bool autoNormal, bool invertNormal, IntPtr mat);

		[DllImport("PathTracer.dll")]
		protected static extern IntPtr ConstructDistortedSphere(Vec3 center, float radius, float frequency, float amplitude, IntPtr mat);

		[DllImport("PathTracer.dll")]
		protected static extern IntPtr ConstructLambertian(float r, float g, float b);

		[DllImport("PathTracer.dll")]
		protected static extern IntPtr ConstructMetal(float r, float g, float b, float fuzz);

		[DllImport("PathTracer.dll")]
		protected static extern IntPtr ConstructDielectric(float r, float g, float b, float refractiveIndex);

		[DllImport("PathTracer.dll")]
		protected static extern IntPtr ConstructDiffuseLight(float r, float g, float b);

		[DllImport("PathTracer.dll")]
		private static extern void DestroyHittable(IntPtr ptr);

		[DllImport("PathTracer.dll")]
		private static extern void DestroyMaterial(IntPtr ptr);
		#endregion
	}
}
