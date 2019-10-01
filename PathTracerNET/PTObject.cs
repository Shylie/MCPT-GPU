using System;
using System.Runtime.InteropServices;
using System.Xml.Serialization;

using PathTracerNET.Hittables;
using PathTracerNET.Hittables.Geometric;
using PathTracerNET.Hittables.Modifier;
using PathTracerNET.Hittables.Plane;
using PathTracerNET.Materials;
using PathTracerNET.Textures;

namespace PathTracerNET
{
	[Serializable]
	[XmlInclude(typeof(ConstantTexture))]
	[XmlInclude(typeof(CheckerboardTexture))]
	[XmlInclude(typeof(NoiseTexture))]
	[XmlInclude(typeof(Dieletric))]
	[XmlInclude(typeof(DiffuseLight))]
	[XmlInclude(typeof(Lambertian))]
	[XmlInclude(typeof(Metal))]
	[XmlInclude(typeof(Sphere))]
	[XmlInclude(typeof(Rotation))]
	[XmlInclude(typeof(Translation))]
	[XmlInclude(typeof(RectangularPlane))]
	[XmlInclude(typeof(TriangularPlane))]
	[XmlInclude(typeof(HittableList))]
	public abstract class PTObject
	{
		~PTObject()
		{
			if (_pointer != IntPtr.Zero) Destroy();
		}

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
					DestroyHittable(_pointer);
					break;
				case PTObjectKind.Material:
					DestroyMaterial(_pointer);
					break;
				case PTObjectKind.Texture:
					DestroyTexture(_pointer);
					break;
			}
			_pointer = IntPtr.Zero;
			Valid = false;
		}
		#endregion

		#region ENUMS
		public enum PTObjectKind
		{
			Invalid,
			Hittable,
			Material,
			Texture
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
		protected static extern IntPtr ConstructConstantTexture(float r, float g, float b);

		[DllImport("PathTracer.dll")]
		protected static extern IntPtr ConstructCheckerboardTexture(IntPtr a, IntPtr b, Vec3 offset, Vec3 frequency);

		[DllImport("PathTracer.dll")]
		protected static extern IntPtr ConstructNoiseTexture(int tiles);

		[DllImport("PathTracer.dll")]
		protected static extern IntPtr ConstructLambertian(IntPtr texture);

		[DllImport("PathTracer.dll")]
		protected static extern IntPtr ConstructMetal(IntPtr texture, float fuzz);

		[DllImport("PathTracer.dll")]
		protected static extern IntPtr ConstructDielectric(IntPtr texture, float refractiveIndex);

		[DllImport("PathTracer.dll")]
		protected static extern IntPtr ConstructDiffuseLight(IntPtr texture);

		[DllImport("PathTracer.dll")]
		private static extern void DestroyHittable(IntPtr ptr);

		[DllImport("PathTracer.dll")]
		private static extern void DestroyMaterial(IntPtr ptr);

		[DllImport("PathTracer.dll")]
		private static extern void DestroyTexture(IntPtr ptr);
		#endregion
	}
}
