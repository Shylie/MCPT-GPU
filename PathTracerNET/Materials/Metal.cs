using System;

namespace PathTracerNET.Materials
{
	[Serializable]
	public sealed class Metal : PTObject
	{
		public Metal() { }

		public Metal(float r, float g, float b, float fuzz)
		{
			R = r;
			G = g;
			B = b;
			Fuzz = fuzz;
		}

		public override PTObjectKind Kind => PTObjectKind.Material;

		internal override IntPtr Init()
		{
			return ConstructMetal(R, G, B, Fuzz);
		}

		public float R { get; set; }
		public float G { get; set; }
		public float B { get; set; }
		public float Fuzz { get; set; }
	}
}
