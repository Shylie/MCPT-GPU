using System;
using System.Xml.Serialization;

namespace PathTracerNET.Materials
{
	[Serializable]
	public sealed class Lambertian : Material
	{
		public Lambertian() { }

		public Lambertian(Texture texture)
		{
			Texture = texture;
		}

		internal override IntPtr Init()
		{
			return ConstructLambertian(Texture.Pointer);
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

		[XmlIgnore]
		private Texture _texture;

		private void TextureInvalidated(PTObject sender) => Destroy();
	}
}
