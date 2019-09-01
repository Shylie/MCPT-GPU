using System;

namespace PathTracerNET
{
	internal class Program
	{
		private const int WIDTH = 1024, HEIGHT = 512, SAMPLES = 250;

		private static void Main(string[] args)
		{
			PTObject lightMaterial = PTObject.DiffuseLight(2f, 2f, 2f);
			PTObject glass = PTObject.Dieletric(1f, 1f, 1f, 1.54f);
			PTObject blueMatte = PTObject.Lambertian(0.4f, 0.7f, 0.9f);

			PTObject light = PTObject.Sphere(new Vec3(0.4f, 3f, -0.4f), 1.8f, lightMaterial);
			PTObject sphere = PTObject.Sphere(new Vec3(0f, -100.1f, 0f), 100f, blueMatte);
			PTObject cube = PTObject.RectangularPrism(-1f, 1f, 0f, 0.25f, -1f, 1f, glass);

			PTObject scene = PTObject.HittableList(light, sphere, cube);

			string fname = (args.Length == 1) ? args[0] : "test";

			PTObject.RenderSceneChunked(WIDTH, HEIGHT, SAMPLES, fname, new Vec3(2f, 0.6f, -2f), new Vec3(-2f, 0f, 2f), new Vec3(0f, 1f, 0f), (float)Math.PI / 3.4f, (float)WIDTH / HEIGHT, scene);

			scene.Destroy();
		}
	}
}
