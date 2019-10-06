namespace PathTracerNET.Expression
{
	internal interface IContext
	{
		double ResolveVariable(string name);
		double CallFunction(string name, double[] args);
	}
}
