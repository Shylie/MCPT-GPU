using System.IO;

namespace PathTracerNET.Expression.Contexts
{
	internal class EmptyContext : IContext
	{
		public double CallFunction(string name, double[] args)
		{
			throw new InvalidDataException($"Unknown function: '{name}'");
		}

		public double ResolveVariable(string name)
		{
			throw new InvalidDataException($"Unknown variable: '{name}'");
		}
	}
}
