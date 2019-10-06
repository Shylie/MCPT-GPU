using System;
using System.Linq;
using System.Xml.Serialization;

namespace PathTracerNET.Hittables
{
	[Serializable]
	public sealed class HittableList : Hittable
	{
		public HittableList() { }

		public HittableList(params Hittable[] hittables)
		{
			Hittables = hittables;
		}

		internal override IntPtr Init()
		{
			return ConstructHittableList(Hittables.Length, Hittables.Select(hittable => hittable.Pointer).ToArray());
		}

		public Hittable[] Hittables
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
		private Hittable[] _hittables;

		internal override void Recalculate(double time)
		{
			base.Recalculate(time);
			foreach (Hittable hittable in _hittables)
			{
				hittable?.Recalculate(time);
			}
		}

		private void HittableInvalidated(PTObject sender) => Destroy();
	}
}
