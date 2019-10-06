namespace PathTracerNET.Expression.Nodes
{
	internal class NumberNode : INode
	{
		public NumberNode(double number)
		{
			_number = number;
		}

		public double Evaluate(IContext context) => _number;

		private readonly double _number;
	}
}
