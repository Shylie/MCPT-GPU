using System;
using System.Xml.Serialization;

namespace PathTracerNET.Textures
{
	public sealed class NoiseTexture : Texture
	{
		public NoiseTexture() { }

		public NoiseTexture(int tiles)
		{
			Tiles = tiles;
		}

		internal override IntPtr Init()
		{
			return ConstructNoiseTexture(Tiles);
		}

		public int Tiles
		{
			get
			{
				return _tiles;
			}
			set
			{
				_tiles = value;
				if (Valid) Destroy();
			}
		}

		[XmlIgnore]
		private int _tiles;
	}
}
