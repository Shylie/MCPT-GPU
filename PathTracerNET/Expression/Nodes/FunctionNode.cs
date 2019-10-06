using System.Linq;

namespace PathTracerNET.Expression.Nodes
{
	internal class FunctionNode : INode
	{
		public FunctionNode(string functionName, INode[] args)
		{
			_functionName = functionName;
			_args = args;
		}

		public double Evaluate(IContext context)
		{
			return context.CallFunction(_functionName, _args.Select(node => node.Evaluate(context)).ToArray());
		}

		private readonly string _functionName;
		private readonly INode[] _args;
	}
}
