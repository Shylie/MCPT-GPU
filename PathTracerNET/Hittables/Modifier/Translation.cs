using System;
using System.Xml.Serialization;

namespace PathTracerNET.Hittables.Modifier
{
	[Serializable]
	public sealed class Translation : Hittable
	{
		public Translation() { }

		public Translation(Vec3 offset, Hittable hittable)
		{
			if (hittable.Kind != PTObjectKind.Hittable) throw new ArgumentException("Invalid PTObjectKind: not a Hittable.", nameof(hittable));

			Offset = offset;
			Hittable = hittable;
		}

		internal override IntPtr Init()
		{
			return ConstructTranslation(Offset, Hittable.Pointer);
		}

		public Vec3 Offset
		{
			get
			{
				return _offset;
			}
			set
			{
				_offset = value;
				if (Valid) Destroy();
			}
		}

		public Hittable Hittable
		{
			get
			{
				return _hittable;
			}
			set
			{
				if (_hittable != null) _hittable.Invalidated -= HittableInvalidated;
				_hittable = value;
				_hittable.Invalidated += HittableInvalidated;
				if (Valid) Destroy();
			}
		}

		[XmlIgnore]
		private Vec3 _offset;

		[XmlIgnore]
		private Hittable _hittable;

		private void HittableInvalidated(PTObject sender) => Destroy();
	}
}
