using System.IO;

namespace PathTracerNET.Expression
{
	internal class Tokenizer
	{
		public Tokenizer(string expression)
		{
			_expression = expression;
			NextChar();
			NextToken();
		}

		public Token Token { get; private set; }

		public double Number { get; private set; }

		public string Identifier { get; private set; }

		public void NextToken()
		{
			while (char.IsWhiteSpace(_currentChar))
			{
				NextChar();
			}

			switch (_currentChar)
			{
				case '\0':
					Token = Token.EOF;
					return;

				case '+':
					NextChar();
					Token = Token.Add;
					return;

				case '-':
					NextChar();
					Token = Token.Subtract;
					return;

				case '*':
					NextChar();
					Token = Token.Multiply;
					return;

				case '/':
					NextChar();
					Token = Token.Divide;
					return;

				case '(':
					NextChar();
					Token = Token.OpenParenthesis;
					return;

				case ')':
					NextChar();
					Token = Token.CloseParenthesis;
					return;

				case ',':
					NextChar();
					Token = Token.Separator;
					return;
			}

			if (char.IsDigit(_currentChar) || _currentChar == '.')
			{
				string numStr = "";
				bool hasDecimalPoint = false;
				while (char.IsDigit(_currentChar) || (!hasDecimalPoint && _currentChar == '.'))
				{
					numStr += _currentChar;
					if (!hasDecimalPoint && _currentChar == '.') hasDecimalPoint = true;
					NextChar();
				}

				Number = double.Parse(numStr);
				Token = Token.Number;
				return;
			}

			if (char.IsLetter(_currentChar) || _currentChar == '_')
			{
				string idStr = "";
				while (char.IsLetter(_currentChar) || _currentChar == '_')
				{
					idStr += _currentChar;
					NextChar();
				}

				Identifier = idStr;
				if (_currentChar == '(')
				{
					Token = Token.FunctionIdentifier;
					NextChar();
				}
				else
				{
					Token = Token.Identifier;
				}
				return;
			}

			throw new InvalidDataException($"Unexpected character at index {_pos}: {_currentChar}");
		}

		private void NextChar()
		{
			_currentChar = _pos >= _expression.Length ? '\0' : _expression[_pos++];
		}

		private string _expression;
		private int _pos = 0;
		private char _currentChar;
	}
}
