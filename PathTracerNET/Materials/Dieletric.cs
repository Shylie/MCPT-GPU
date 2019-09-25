using System;
using System.Xml.Serialization;

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

		public float R
		{
			get
			{
				return _r;
			}
			set
			{
				_r = value;
				if (Valid) Destroy();
			}
		}

		public float G
		{
			get
			{
				return _g;
			}
			set
			{
				_g = value;
				if (Valid) Destroy();
			}
		}

		public float B
		{
			get
			{
				return _b;
			}
			set
			{
				_b = value;
				if (Valid) Destroy();
			}
		}

		public float RefractiveIndex
		{
			get
			{
				return _refractiveIndex;
			}
			set
			{
				_refractiveIndex = value;
				if (Valid) Destroy();
			}
		}

		[XmlIgnore]
		private float _r, _g, _b, _refractiveIndex;
	}
}
