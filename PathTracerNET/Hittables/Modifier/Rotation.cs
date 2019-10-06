using System;
using System.Xml.Serialization;

using PathTracerNET.Expression;
using PathTracerNET.Expression.Contexts;

namespace PathTracerNET.Hittables.Modifier
{
	[Serializable]
	public sealed class Rotation : Hittable
	{
		public Rotation() { }

		internal override IntPtr Init()
		{
			return ConstructRotation(Theta, Axis, Hittable.Pointer);
		}

		public string ThetaExpression
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
		private float Theta
		{
			get
			{
				return (float)Parser.Evaluate(ThetaExpression, new AnimationContext(Time));
			}
		}

		[XmlIgnore]
		private string _theta;

		[XmlIgnore]
		private Alignment _axis;

		[XmlIgnore]
		private Hittable _hittable;

		internal override void Recalculate(double time)
		{
			base.Recalculate(time);
			_hittable?.Recalculate(time);
		}

		private void HittableInvalidated(PTObject sender) => Destroy();
	}
}
