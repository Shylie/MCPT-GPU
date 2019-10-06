using System;
using System.Xml.Serialization;

using PathTracerNET.Expression;
using PathTracerNET.Expression.Contexts;

namespace PathTracerNET.Hittables.Geometric
{
	[Serializable]
	public sealed class Sphere : Hittable
	{
		public Sphere() { }

		internal override IntPtr Init()
		{
			return ConstructSphere(Center, Radius, Material.Pointer);
		}

		public string CenterXExpression
		{
			get
			{
				return _cx;
			}
			set
			{
				_cx = value;
				if (Valid) Destroy();
			}
		}

		public string CenterYExpression
		{
			get
			{
				return _cy;
			}
			set
			{
				_cy = value;
				if (Valid) Destroy();
			}
		}

		public string CenterZExpression
		{
			get
			{
				return _cz;
			}
			set
			{
				_cz = value;
				if (Valid) Destroy();
			}
		}

		public string RadiusExpression
		{
			get
			{
				return _r;
			}
			set
			{
				_r = value;
				if (Valid) Destroy();
			}
		}

		public Material Material
		{
			get
			{
				return _material;
			}
			set
			{
				if (_material != null) _material.Invalidated -= MaterialInvalidated;
				_material = value;
				_material.Invalidated += MaterialInvalidated;
				if (Valid) Destroy();
			}
		}

		[XmlIgnore]
		private Vec3 Center
		{
			get
			{
				AnimationContext context = new AnimationContext(Time);
				float x = (float)Parser.Evaluate(CenterXExpression, context);
				float y = (float)Parser.Evaluate(CenterYExpression, context);
				float z = (float)Parser.Evaluate(CenterZExpression, context);
				return new Vec3(x, y, z);
			}
		}

		[XmlIgnore]
		private float Radius
		{
			get
			{
				AnimationContext context = new AnimationContext(Time);
				return (float)Parser.Evaluate(RadiusExpression, context);
			}
		}

		[XmlIgnore]
		private Material _material;

		[XmlIgnore]
		private string _cx, _cy, _cz, _r;

		internal override void Recalculate(double time)
		{
			base.Recalculate(time);
			_material?.Recalculate(time);
		}

		private void MaterialInvalidated(PTObject sender) => Destroy();
	}
}
