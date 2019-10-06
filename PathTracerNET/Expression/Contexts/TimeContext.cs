using System.IO;

namespace PathTracerNET.Expression.Contexts
{
	internal class AnimationContext : IContext
	{
		public AnimationContext(double time)
		{
			_time = time;
		}

		public double CallFunction(string name, double[] args)
		{
			switch (name)
			{
				case "sin":
					if (args.Length != 1) throw new InvalidDataException($"Invalid arguments for function '{name}'");
					return System.Math.Sin(args[0]);

				case "cos":
					if (args.Length != 1) throw new InvalidDataException($"Invalid arguments for function '{name}'");
					return System.Math.Cos(args[0]);

				case "tan":
					if (args.Length != 1) throw new InvalidDataException($"Invalid arguments for function '{name}'");
					return System.Math.Tan(args[0]);

				case "pow":
					if (args.Length != 2) throw new InvalidDataException($"Invalid arguments for function '{name}'");
					return System.Math.Pow(args[0], args[1]);

				case "log":
					if (args.Length != 2) throw new InvalidDataException($"Invalid arguments for function '{name}'");
					return System.Math.Log(args[0], args[1]);

				default:
					throw new InvalidDataException($"Unknown function: '{name}'");
			}
		}

		public double ResolveVariable(string name)
		{
			switch (name)
			{
				case "time":
					return _time;

				default:
					throw new InvalidDataException($"Unknown variable: '{name}'");
			}
		}

		private readonly double _time;
	}
}
