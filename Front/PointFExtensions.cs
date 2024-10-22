using System.Drawing;

public static class PointFExtensions
{
	public static Point ToPoint(this PointF point)
	{
		return new Point((int)(point.X), (int)(point.Y));
	}
}