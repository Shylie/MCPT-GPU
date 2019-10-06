using System;

namespace PathTracerNET.Expression.Nodes
{
	internal class BinaryNode : INode
	{
		public BinaryNode(INode lhs, INode rhs, Func<double, double, double> op)
		{
			_lhs = lhs;
			_rhs = rhs;
			_op = op;
		}

		public double Evaluate(IContext context) => _op(_lhs.Evaluate(context), _rhs.Evaluate(context));

		private readonly INode _lhs, _rhs;
		private readonly Func<double, double, double> _op;
	}
}
