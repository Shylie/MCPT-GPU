using System;
using System.Runtime.Serialization;

namespace PathTracerNET.Hittables.Plane
{
	[Serializable]
	public sealed class TriangularPlane : PTObject
	{
		public TriangularPlane() { }

		public TriangularPlane(float a1, float b1, float a2, float b2, float a3, float b3, float k, Alignment plane, bool autoNormal, bool invertNormal, PTObject material)
		{
			switch (plane)
			{
				case Alignment.None:
				case Alignment.X:
				case Alignment.Y:
				case Alignment.Z:
				case Alignment.X | Alignment.Y | Alignment.Z:
					throw new ArgumentException("Invalid alignment: must be XY, XZ, or YZ.", nameof(plane));
			}
			if (material.Kind != PTObjectKind.Material) throw new ArgumentException("Invalid PTObjectKind: not a Material.", nameof(material));

			A1 = a1;
			B1 = b1;
			A2 = a2;
			B2 = b2;
			A3 = a3;
			B3 = b3;
			K = k;
			Plane = plane;
			AutoNormal = autoNormal;
			InvertNormal = invertNormal;
			Material = material;
		}

		public override PTObjectKind Kind => PTObjectKind.Hittable;

		internal override IntPtr Init()
		{
			return ConstructTriangularPlane(A1, B1, A2, B2, A3, B3, K, Plane, AutoNormal, InvertNormal, Material.Pointer);
		}

		public float A1 { get; set; }
		public float B1 { get; set; }
		public float A2 { get; set; }
		public float B2 { get; set; }
		public float A3 { get; set; }
		public float B3 { get; set; }
		public float K { get; set; }
		public Alignment Plane { get; set; }
		public bool AutoNormal { get; set; }
		public bool InvertNormal { get; set; }
		public PTObject Material { get; set; }

		[OnDeserialized]
		private void OnDeserialized(StreamingContext context)
		{
			if (!Valid)
			{
				Pointer = ConstructTriangularPlane(A1, B1, A2, B2, A3, B3, K, Plane, AutoNormal, InvertNormal, Material.Pointer);
				Valid = true;
			}
		}
	}
}
