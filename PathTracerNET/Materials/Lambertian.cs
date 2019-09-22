using System;

namespace PathTracerNET.Materials
{
	[Serializable]
	public sealed class Lambertian : PTObject
	{
		public Lambertian() { }

		public Lambertian(float r, float g, float b)
		{
			R = r;
			G = g;
			B = b;
		}

		public override PTObjectKind Kind => PTObjectKind.Material;

		internal override IntPtr Init()
		{
			return ConstructLambertian(R, G, B);
		}

		public float R { get; set; }
		public float G { get; set; }
		public float B { get; set; }
	}
}
