using System;
using System.Xml.Serialization;

namespace PathTracerNET.Materials
{
	[Serializable]
	public sealed class Metal : Material
	{
		public Metal() { }

		public Metal(Texture texture, float fuzz)
		{
			Texture = texture;
			Fuzz = fuzz;
		}

		internal override IntPtr Init()
		{
			return ConstructMetal(Texture.Pointer, Fuzz);
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
		private Texture _texture;

		[XmlIgnore]
		private float _fuzz;

		private void TextureInvalidated(PTObject sender) => Destroy();
	}
}
