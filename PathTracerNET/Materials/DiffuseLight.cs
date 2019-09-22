using System;

namespace PathTracerNET.Materials
{
	[Serializable]
	public sealed class DiffuseLight : PTObject
	{
		public DiffuseLight() { }

		public DiffuseLight(float r, float g, float b)
		{
			R = r;
			G = g;
			B = b;
		}

		public override PTObjectKind Kind => PTObjectKind.Material;

		internal override IntPtr Init()
		{
			return ConstructDiffuseLight(R, G, B);
		}

		public float R { get; set; }
		public float G { get; set; }
		public float B { get; set; }
	}
}
