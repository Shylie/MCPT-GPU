﻿using System;
using System.Xml.Serialization;

namespace PathTracerNET.Hittables.Plane
{
	[Serializable]
	public sealed class TriangularPlane : Hittable
	{
		public TriangularPlane() { }

		public TriangularPlane(float a1, float b1, float a2, float b2, float a3, float b3, float k, Alignment plane, bool autoNormal, bool invertNormal, Material material)
		{
			switch (plane)
			{
				case Alignment.None:
				case Alignment.X:
				case Alignment.Y:
				case Alignment.Z:
				case Alignment.X | Alignment.Y | Alignment.Z:
					throw new ArgumentException("Invalid alignment: must be XY, XZ, or YZ.", nameof(plane));
			}

			A1 = a1;
			B1 = b1;
			A2 = a2;
			B2 = b2;
			A3 = a3;
			B3 = b3;
			K = k;
			Plane = plane;
			AutoNormal = autoNormal;
			InvertNormal = invertNormal;
			Material = material;
		}

		internal override IntPtr Init()
		{
			return ConstructTriangularPlane(A1, B1, A2, B2, A3, B3, K, Plane, AutoNormal, InvertNormal, Material.Pointer);
		}

		public float A1
		{
			get
			{
				return _a1;
			}
			set
			{
				_a1 = value;
				if (Valid) Destroy();
			}
		}

		public float B1
		{
			get
			{
				return _b1;
			}
			set
			{
				_b1 = value;
				if (Valid) Destroy();
			}
		}

		public float A2
		{
			get
			{
				return _a2;
			}
			set
			{
				_a2 = value;
				if (Valid) Destroy();
			}
		}

		public float B2
		{
			get
			{
				return _b2;
			}
			set
			{
				_b2 = value;
				if (Valid) Destroy();
			}
		}

		public float A3
		{
			get
			{
				return _a3;
			}
			set
			{
				_a3 = value;
				if (Valid) Destroy();
			}
		}

		public float B3
		{
			get
			{
				return _b3;
			}
			set
			{
				_b3 = value;
				if (Valid) Destroy();
			}
		}

		public float K
		{
			get
			{
				return _k;
			}
			set
			{
				_k = value;
				if (Valid) Destroy();
			}
		}

		public Alignment Plane
		{
			get
			{
				return _plane;
			}
			set
			{
				_plane = value;
				if (Valid) Destroy();
			}
		}

		public bool AutoNormal
		{
			get
			{
				return _autoNormal;
			}
			set
			{
				_autoNormal = value;
				if (Valid) Destroy();
			}
		}

		public bool InvertNormal
		{
			get
			{
				return _invertNormal;
			}
			set
			{
				_invertNormal = value;
				if (Valid) Destroy();
			}
		}

		public Material Material
		{
			get
			{
				return _material;
			}
			set
			{
				_material.Invalidated -= MaterialInvalidated;
				_material = value;
				_material.Invalidated += MaterialInvalidated;
				if (Valid) Destroy();
			}
		}

		[XmlIgnore]
		private float _a1, _b1, _a2, _b2, _a3, _b3, _k;

		[XmlIgnore]
		private Alignment _plane;

		[XmlIgnore]
		private bool _autoNormal, _invertNormal;

		[XmlIgnore]
		private Material _material;

		private void MaterialInvalidated(PTObject sender) => Destroy();
	}
}
