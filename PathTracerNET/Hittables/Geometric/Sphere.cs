using System;
using System.Xml.Serialization;

namespace PathTracerNET.Hittables.Geometric
{
	[Serializable]
	public sealed class Sphere : Hittable
	{
		public Sphere() { }

		public Sphere(Vec3 center, float radius, Material material)
		{
			Center = center;
			Radius = radius;
			Material = material;
		}

		internal override IntPtr Init()
		{
			return ConstructSphere(Center, Radius, Material.Pointer);
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

		public Material Material
		{
			get
			{
				return _material;
			}
			set
			{
				if (_material != null) _material.Invalidated -= MaterialInvalidated;
				_material = value;
				_material.Invalidated += MaterialInvalidated;
				if (Valid) Destroy();
			}
		}

		[XmlIgnore]
		private Vec3 _center;

		[XmlIgnore]
		private float _radius;

		[XmlIgnore]
		private Material _material;

		private void MaterialInvalidated(PTObject sender) => Destroy();
	}
}
