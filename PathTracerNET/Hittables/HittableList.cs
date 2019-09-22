using System;
using System.Linq;

namespace PathTracerNET.Hittables
{
	[Serializable]
	public sealed class HittableList : PTObject
	{
		public HittableList() { }

		public HittableList(params PTObject[] hittables)
		{
			foreach (PTObject hittable in hittables)
			{
				if (hittable.Kind != PTObjectKind.Hittable) throw new ArgumentException("Invalid PTObjectKind: not a Hittable.");
			}
			Hittables = hittables;
		}

		public override PTObjectKind Kind => PTObjectKind.Hittable;

		internal override IntPtr Init()
		{
			return ConstructHittableList(Hittables.Length, Hittables.Select(hittable => hittable.Pointer).ToArray());
		}

		public PTObject[] Hittables { get; set; }
	}
}
