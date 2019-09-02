using System;

namespace PathTracerNET
{
	internal class Program
	{
		private const int WIDTH = 1024, HEIGHT = 512, SAMPLES = 500;

		private static void Main(string[] args)
		{
			PTObject lightMaterial = PTObject.DiffuseLight(3f, 3f, 3f);
			PTObject glass = PTObject.Dieletric(1f, 1f, 1f, 1.2f);
			PTObject blueMatte = PTObject.Lambertian(0.4f, 0.7f, 0.9f);
			PTObject yellowMatte = PTObject.Lambertian(0.8f, 0.7f, 0.4f);
			PTObject redMirror = PTObject.Metal(0.99f, 0.7f, 0.4f, 0.08f);

			PTObject light = PTObject.Sphere(new Vec3(0f, 3f, 0f), 1.7f, lightMaterial);
			PTObject sphere = PTObject.Sphere(new Vec3(0f, -100.3f, 0f), 100f, blueMatte);
			PTObject cube = PTObject.RectangularPrism(-1f, 1f, 0.2f, 0.45f, -1f, 1f, glass);
			PTObject otherCube = PTObject.RectangularPrism(0f, 0.4f, 0f, 0.4f, 0f, 0.4f, redMirror);
			PTObject rotatedCube = PTObject.Rotation((float)Math.PI / 10.4f, PTObject.Alignment.X, otherCube);
			rotatedCube = PTObject.Rotation((float)Math.PI / 12.8f, PTObject.Alignment.Z, rotatedCube);
			PTObject translatedRotatedCube = PTObject.Translation(new Vec3(-0.4f, 0.75f, -0.4f), rotatedCube);
			PTObject otherSphere = PTObject.Sphere(new Vec3(0f, 0.3f, 0f), 0.4f, glass);
			PTObject triangle = PTObject.TriangularPlane(-1f, -1f, 0f, 1f, 1f, -1f, -0.15f, PTObject.Alignment.X | PTObject.Alignment.Z, true, false, yellowMatte);

			PTObject scene = PTObject.HittableList(light, sphere, otherSphere, cube, translatedRotatedCube, triangle);

			string fname = (args.Length == 1) ? args[0] : "test";

			PTObject.RenderSceneChunked(WIDTH, HEIGHT, SAMPLES, fname, new Vec3(2f, 1.2f, -2f), new Vec3(-2f, 0f, 2f), new Vec3(0f, 1f, 0f), (float)Math.PI / 3f, (float)WIDTH / HEIGHT, scene);

			scene.Destroy();
			cube.Destroy();
		}
	}
}
