using System;

namespace PathTracerNET.Hittables.Modifier
{
	[Serializable]
	public sealed class Translation : PTObject
	{
		public Translation() { }

		public Translation(Vec3 offset, PTObject hittable)
		{
			if (hittable.Kind != PTObjectKind.Hittable) throw new ArgumentException("Invalid PTObjectKind: not a Hittable.", nameof(hittable));

			Offset = offset;
			Hittable = hittable;
		}

		public override PTObjectKind Kind => PTObjectKind.Hittable;

		internal override IntPtr Init()
		{
			return ConstructTranslation(Offset, Hittable.Pointer);
		}

		public Vec3 Offset { get; set; }
		public PTObject Hittable { get; set; }
	}
}
