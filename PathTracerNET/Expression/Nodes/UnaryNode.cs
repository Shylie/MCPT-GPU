using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PathTracerNET.Expression.Nodes
{
	internal class UnaryNode : INode
	{
		public UnaryNode(INode node, Func<double, double> op)
		{
			_node = node;
			_op = op;
		}

		public double Evaluate(IContext context) => _op(_node.Evaluate(context));

		private readonly INode _node;
		private readonly Func<double, double> _op;
	}
}
