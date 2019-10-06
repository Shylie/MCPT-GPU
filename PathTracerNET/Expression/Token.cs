namespace PathTracerNET.Expression
{
	internal enum Token
	{
		EOF,
		Number,
		Add,
		Subtract,
		Multiply,
		Divide,
		OpenParenthesis,
		CloseParenthesis,
		Identifier,
		FunctionIdentifier,
		Separator
	}
}
