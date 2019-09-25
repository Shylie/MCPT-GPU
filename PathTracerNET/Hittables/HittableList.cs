using System;
using System.Linq;
using System.Xml.Serialization;

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

		public PTObject[] Hittables
		{
			get
			{
				return _hittables;
			}
			set
			{
				if (_hittables != null)
				{
					for (int i = 0; i < _hittables.Length; i++)
					{
						if (_hittables[i] != null)
						{
							_hittables[i].Invalidated -= HittableInvalidated;
						}
					}
				}
				_hittables = value;
				for (int i = 0; i < _hittables.Length; i++)
				{
					_hittables[i].Invalidated += HittableInvalidated;
				}
				if (Valid) Destroy();
			}
		}

		[XmlIgnore]
		private PTObject[] _hittables;

		private void HittableInvalidated(PTObject sender) => Destroy();
	}
}
