using System;

namespace PathTracerNET.Hittables.Modifier
{
	[Serializable]
	public sealed class Rotation : PTObject
	{
		public Rotation() { }

		public Rotation(float theta, Alignment axis, PTObject hittable)
		{
			if (axis != Alignment.X && axis != Alignment.Y && axis != Alignment.Z) throw new ArgumentException("Invalid Alignment: must be X, Y, or Z.", nameof(axis));
			if (hittable.Kind != PTObjectKind.Hittable) throw new ArgumentException("Invalid PTObjectKind: not a Hittable.", nameof(hittable));

			Theta = theta;
			Axis = axis;
			Hittable = hittable;
		}

		public override PTObjectKind Kind => PTObjectKind.Hittable;

		internal override IntPtr Init()
		{
			return ConstructRotation(Theta, Axis, Hittable.Pointer);
		}

		public float Theta { get; set; }
		public Alignment Axis { get; set; }
		public PTObject Hittable { get; set; }
	}
}
