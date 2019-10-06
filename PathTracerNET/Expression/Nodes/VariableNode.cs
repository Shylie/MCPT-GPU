using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PathTracerNET.Expression.Nodes
{
	internal class VariableNode : INode
	{
		public VariableNode(string variableName)
		{
			_variableName = variableName;
		}

		public double Evaluate(IContext context) => context.ResolveVariable(_variableName);

		private readonly string _variableName;
	}
}
