using System;
using System.Xml.Serialization;

namespace PathTracerNET.Materials
{
	[Serializable]
	public sealed class Dieletric : Material
	{
		public Dieletric() { }

		public Dieletric(Texture texture, float refractiveIndex)
		{
			Texture = texture;
			RefractiveIndex = refractiveIndex;
		}

		internal override IntPtr Init()
		{
			return ConstructDielectric(Texture.Pointer, RefractiveIndex);
		}

		public Texture Texture
		{
			get
			{
				return _texture;
			}
			set
			{
				if (_texture != null) _texture.Invalidated -= TextureInvalidated;
				_texture = value;
				_texture.Invalidated += TextureInvalidated;
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
		private float _refractiveIndex;

		[XmlIgnore]
		private Texture _texture;

		private void TextureInvalidated(PTObject sender) => Destroy();
	}
}
