namespace PathTracerNET.Expression
{
	internal interface INode
	{
		double Evaluate(IContext context);
	}
}
