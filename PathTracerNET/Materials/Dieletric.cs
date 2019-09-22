using System;

namespace PathTracerNET.Materials
{
	[Serializable]
	public sealed class Dieletric : PTObject
	{
		public Dieletric() { }

		public Dieletric(float r, float g, float b, float refractiveIndex)
		{
			R = r;
			G = g;
			B = b;
			RefractiveIndex = refractiveIndex;
		}

		public override PTObjectKind Kind => PTObjectKind.Material;

		internal override IntPtr Init()
		{
			return ConstructDielectric(R, G, B, RefractiveIndex);
		}

		public float R { get; set; }
		public float G { get; set; }
		public float B { get; set; }
		public float RefractiveIndex { get; set; }
	}
}
