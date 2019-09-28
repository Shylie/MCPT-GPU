using System;
using System.Xml.Serialization;

namespace PathTracerNET.Textures
{
	public sealed class CheckerboardTexture : Texture
	{
		public CheckerboardTexture() { }

		public CheckerboardTexture(Texture a, Texture b, Vec3 offset, Vec3 frequency)
		{
			A = a;
			B = b;
			Offset = offset;
			Frequency = frequency;
		}

		internal override IntPtr Init()
		{
			return ConstructCheckerboardTexture(A.Pointer, B.Pointer, Offset, Frequency);
		}

		public Texture A
		{
			get
			{
				return _a;
			}
			set
			{
				if (_a != null) _a.Invalidated -= TextureInvalidated;
				_a = value;
				_a.Invalidated += TextureInvalidated;
				if (Valid) Destroy();
			}
		}

		public Texture B
		{
			get
			{
				return _b;
			}
			set
			{
				if (_b != null) _b.Invalidated -= TextureInvalidated;
				_b = value;
				_b.Invalidated += TextureInvalidated;
				if (Valid) Destroy();
			}
		}

		public Vec3 Offset
		{
			get
			{
				return _offset;
			}
			set
			{
				_offset = value;
				if (Valid) Destroy();
			}
		}

		public Vec3 Frequency
		{
			get
			{
				return _frequency;
			}
			set
			{
				_frequency = value;
				if (Valid) Destroy();
			}
		}

		[XmlIgnore]
		private Texture _a, _b;

		[XmlIgnore]
		private Vec3 _offset, _frequency;

		private void TextureInvalidated(PTObject sender) => Destroy();
	}
}
