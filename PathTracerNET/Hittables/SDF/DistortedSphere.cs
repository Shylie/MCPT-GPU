using System;

namespace PathTracerNET.Hittables.SDF
{
	[Serializable]
	public sealed class DistortedSphere : PTObject
	{
		public DistortedSphere() { }

		public DistortedSphere(Vec3 center, float radius, float frequency, float amplitude, PTObject material)
		{
			if (material.Kind != PTObjectKind.Material) throw new ArgumentException("Invalid PTObjectKind: not a Material.", nameof(material));

			Center = center;
			Radius = radius;
			Frequency = frequency;
			Amplitude = amplitude;
			Material = material;
		}

		public override PTObjectKind Kind => PTObjectKind.Hittable;

		internal override IntPtr Init()
		{
			return ConstructDistortedSphere(Center, Radius, Frequency, Amplitude, Material.Pointer);
		}

		public Vec3 Center { get; set; }
		public float Radius { get; set; }
		public float Frequency { get; set; }
		public float Amplitude { get; set; }
		public PTObject Material { get; set; }
	}
}
