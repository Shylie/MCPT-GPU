using System;
using System.Xml.Serialization;

namespace PathTracerNET.Hittables.SDF
{
	[Serializable]
	public sealed class DistortedSphere : Hittable
	{
		public DistortedSphere() { }

		public DistortedSphere(Vec3 center, float radius, float frequency, float amplitude, Material material)
		{
			Center = center;
			Radius = radius;
			Frequency = frequency;
			Amplitude = amplitude;
			Material = material;
		}

		internal override IntPtr Init()
		{
			return ConstructDistortedSphere(Center, Radius, Frequency, Amplitude, Material.Pointer);
		}

		public Vec3 Center
		{
			get
			{
				return _center;
			}
			set
			{
				_center = value;
				if (Valid) Destroy();
			}
		}

		public float Radius
		{
			get
			{
				return _radius;
			}
			set
			{
				_radius = value;
				if (Valid) Destroy();
			}
		}

		public float Frequency
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

		public float Amplitude
		{
			get
			{
				return _amplitude;
			}
			set
			{
				_amplitude = value;
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
		private Vec3 _center;

		[XmlIgnore]
		private float _radius, _frequency, _amplitude;

		[XmlIgnore]
		private Material _material;

		private void MaterialInvalidated(PTObject sender) => Destroy();
	}
}
