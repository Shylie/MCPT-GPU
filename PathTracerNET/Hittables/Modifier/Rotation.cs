using System;
using System.Xml.Serialization;

namespace PathTracerNET.Hittables.Modifier
{
	[Serializable]
	public sealed class Rotation : Hittable
	{
		public Rotation() { }

		public Rotation(float theta, Alignment axis, Hittable hittable)
		{
			if (axis != Alignment.X && axis != Alignment.Y && axis != Alignment.Z) throw new ArgumentException("Invalid Alignment: must be X, Y, or Z.", nameof(axis));

			Theta = theta;
			Axis = axis;
			Hittable = hittable;
		}

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
		private float _theta;

		[XmlIgnore]
		private Alignment _axis;

		[XmlIgnore]
		private Hittable _hittable;

		private void HittableInvalidated(PTObject sender) => Destroy();
	}
}
