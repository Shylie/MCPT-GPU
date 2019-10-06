using PathTracerNET.Expression.Contexts;
using PathTracerNET.Expression.Nodes;
using System;
using System.Collections.Generic;

namespace PathTracerNET.Expression
{
	internal class Parser
	{
		public Parser(string expression)
		{
			_tokenizer = new Tokenizer(expression);
		}

		public double Evaluate(IContext context) => (node ?? (node = ParseExpression())).Evaluate(context);
		public double Evaluate() => (node ?? (node = ParseExpression())).Evaluate(new EmptyContext());

		private INode ParseExpression()
		{
			INode node = ParseAddSubtract();

			if (_tokenizer.Token != Token.EOF) throw new Exception("Unexpected characters at end of expression");

			return node;
		}

		private INode ParseAddSubtract()
		{
			INode lhs = ParseMultiplyDivide();

			while (true)
			{
				Func<double, double, double> op = null;
				
				switch (_tokenizer.Token)
				{
					case Token.Add:
						op = (a, b) => a + b;
						break;

					case Token.Subtract:
						op = (a, b) => a - b;
						break;

					default:
						return lhs;
				}

				_tokenizer.NextToken();

				INode rhs = ParseMultiplyDivide();

				lhs = new BinaryNode(lhs, rhs, op);
			}
		}

		private INode ParseMultiplyDivide()
		{
			INode lhs = ParseUnary();

			while (true)
			{
				Func<double, double, double> op = null;

				switch (_tokenizer.Token)
				{
					case Token.Multiply:
						op = (a, b) => a * b;
						break;

					case Token.Divide:
						op = (a, b) => a / b;
						break;

					default:
						return lhs;
				}

				_tokenizer.NextToken();

				INode rhs = ParseUnary();

				lhs = new BinaryNode(lhs, rhs, op);
			}
		}

		private INode ParseUnary()
		{
			if (_tokenizer.Token == Token.Add)
			{
				_tokenizer.NextToken();
				return ParseUnary();
			}

			if (_tokenizer.Token == Token.Subtract)
			{
				_tokenizer.NextToken();

				INode rhs = ParseUnary();

				return new UnaryNode(rhs, (a) => -a);
			}

			return ParseLeaf();
		}

		private INode ParseLeaf()
		{
			if (_tokenizer.Token == Token.Number)
			{
				INode node = new NumberNode(_tokenizer.Number);
				_tokenizer.NextToken();
				return node;
			}

			if (_tokenizer.Token == Token.Identifier)
			{
				INode node = new VariableNode(_tokenizer.Identifier);
				_tokenizer.NextToken();
				return node;
			}

			if (_tokenizer.Token == Token.FunctionIdentifier)
			{
				string name = _tokenizer.Identifier;
				List<INode> pars = new List<INode>();
				_tokenizer.NextToken();
				do
				{
					pars.Add(ParseAddSubtract());
					if (_tokenizer.Token != Token.Separator && _tokenizer.Token != Token.CloseParenthesis) throw new Exception($"Unexpected token: {_tokenizer.Token}");
					_tokenizer.NextToken();
				}
				while (_tokenizer.Token == Token.Separator);

				return new FunctionNode(name, pars.ToArray());
			}

			if (_tokenizer.Token == Token.OpenParenthesis)
			{
				_tokenizer.NextToken();

				INode node = ParseAddSubtract();

				if (_tokenizer.Token != Token.CloseParenthesis) throw new Exception("Missing closing parenthesis");

				_tokenizer.NextToken();

				return node;
			}

			throw new Exception($"Unexpected token: {_tokenizer.Token}");
		}

		private readonly Tokenizer _tokenizer;
		private INode node;

		public static double Evaluate(string expression) => new Parser(expression).Evaluate();
		public static double Evaluate(string expression, IContext context) => new Parser(expression).Evaluate(context);
	}
}
