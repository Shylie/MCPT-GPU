using System;

namespace PathTracerNET.Hittables.Geometric
{
	[Serializable]
	public sealed class Sphere : PTObject
	{
		public Sphere() { }

		public Sphere(Vec3 center, float radius, PTObject material)
		{
			if (radius <= 0f) throw new ArgumentException("Negative or zero radius not allowed.", nameof(radius));
			if (material.Kind != PTObjectKind.Material) throw new ArgumentException("Invalid PTObjectKind: not a Material.", nameof(material));

			Center = center;
			Radius = radius;
			Material = material;
		}

		public override PTObjectKind Kind => PTObjectKind.Hittable;

		internal override IntPtr Init()
		{
			return ConstructSphere(Center, Radius, Material.Pointer);
		}

		public Vec3 Center { get; set; }
		public float Radius { get; set; }
		public PTObject Material { get; set; }
	}
}
