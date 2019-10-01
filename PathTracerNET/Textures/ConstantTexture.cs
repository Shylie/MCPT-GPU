using System;
using System.Xml.Serialization;

namespace PathTracerNET.Textures
{
	public sealed class ConstantTexture : Texture
	{
		public ConstantTexture() { }

		public ConstantTexture(float r, float g, float b)
		{
			R = r;
			G = g;
			B = b;
		}

		internal override IntPtr Init()
		{
			return ConstructConstantTexture(R, G, B);
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

		[XmlIgnore]
		private float _r, _g, _b;
	}
}
