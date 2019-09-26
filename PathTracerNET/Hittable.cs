namespace PathTracerNET
{
	public abstract class Hittable : PTObject
	{
		public sealed override PTObjectKind Kind => PTObjectKind.Hittable;
	}
}
