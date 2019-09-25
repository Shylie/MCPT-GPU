using System;
using System.Xml.Serialization;

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

		public float Theta
		{
			get
			{
				return _theta;
			}
			set
			{
				_theta = value;
				if (Valid) Destroy();
			}
		}

		public Alignment Axis
		{
			get
			{
				return _axis;
			}
			set
			{
				_axis = value;
				if (Valid) Destroy();
			}
		}

		public PTObject Hittable
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
		private float _theta;

		[XmlIgnore]
		private Alignment _axis;

		[XmlIgnore]
		private PTObject _hittable;

		private void HittableInvalidated(PTObject sender) => Destroy();
	}
}
