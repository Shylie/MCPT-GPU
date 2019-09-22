using System;

namespace PathTracerNET.Hittables.Plane
{
	[Serializable]
	public sealed class RectangularPlane : PTObject
	{
		public RectangularPlane() { }

		public RectangularPlane(float a1, float a2, float b1, float b2, float k, Alignment plane, bool autoNormal, bool invertNormal, PTObject material)
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
			A2 = a2;
			B1 = b1;
			B2 = b2;
			K = k;
			Plane = plane;
			AutoNormal = autoNormal;
			InvertNormal = invertNormal;
			Material = material;
		}

		public override PTObjectKind Kind => PTObjectKind.Hittable;

		internal override IntPtr Init()
		{
			return ConstructRectangularPlane(A1, A2, B1, B2, K, Plane, AutoNormal, InvertNormal, Material.Pointer);
		}

		public float A1 { get; set; }
		public float A2 { get; set; }
		public float B1 { get; set; }
		public float B2 { get; set; }
		public float K { get; set; }
		public Alignment Plane { get; set; }
		public bool AutoNormal { get; set; }
		public bool InvertNormal { get; set; }
		public PTObject Material { get; set; }
	}
}
