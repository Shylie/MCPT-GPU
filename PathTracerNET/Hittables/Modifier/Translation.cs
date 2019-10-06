using System;
using System.Xml.Serialization;

using PathTracerNET.Expression;
using PathTracerNET.Expression.Contexts;

namespace PathTracerNET.Hittables.Modifier
{
	[Serializable]
	public sealed class Translation : Hittable
	{
		public Translation() { }

		internal override IntPtr Init()
		{
			return ConstructTranslation(Offset, Hittable.Pointer);
		}

		public string OffsetXExpression
		{
			get
			{
				return _ox;
			}
			set
			{
				_ox = value;
				if (Valid) Destroy();
			}
		}

		public string OffsetYExpression
		{
			get
			{
				return _oy;
			}
			set
			{
				_oy = value;
				if (Valid) Destroy();
			}
		}

		public string OffsetZExpression
		{
			get
			{
				return _oz;
			}
			set
			{
				_oz = value;
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
		private Vec3 Offset
		{
			get
			{
				AnimationContext context = new AnimationContext(Time);
				float x = (float)Parser.Evaluate(OffsetXExpression, context);
				float y = (float)Parser.Evaluate(OffsetYExpression, context);
				float z = (float)Parser.Evaluate(OffsetZExpression, context);
				return new Vec3(x, y, z);
			}
		}

		[XmlIgnore]
		private string _ox, _oy, _oz;

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
