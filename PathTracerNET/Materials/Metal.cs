using System;
using System.Xml.Serialization;

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

		public float Fuzz
		{
			get
			{
				return _fuzz;
			}
			set
			{
				_fuzz = value;
				if (Valid) Destroy();
			}
		}

		[XmlIgnore]
		private float _r, _g, _b, _fuzz;
	}
}
