using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace PathTracerGUI
{
	internal partial class Main : Form
	{

		private Dictionary<string, PTObject> materials = new Dictionary<string, PTObject>();
		private Dictionary<string, PTObject> objects = new Dictionary<string, PTObject>();

		public void AddMaterial(string name, PTObject material)
		{
			materials.Add(name, material);
			listbxMaterials.Items.Add(name);
		}

		public PTObject GetMaterial(string name)
		{
			return materials[name];
		}

		public void AddObject(string name, PTObject obj)
		{
			objects.Add(name, obj);
			listbxObjects.Items.Add(name);
		}

		public PTObject GetObject(string name)
		{
			return objects[name];
		}

		public Main()
		{
			InitializeComponent();

			AddMaterial("lightMaterial", PTObject.DiffuseLight(3f, 3f, 3f));
			AddMaterial("glass", PTObject.Dieletric(1f, 1f, 1f, 1.2f));
			AddMaterial("blueMatte", PTObject.Lambertian(0.4f, 0.7f, 0.9f));
			AddMaterial("yellowMatte", PTObject.Lambertian(0.8f, 0.7f, 0.4f));
			AddMaterial("redMirror", PTObject.Metal(0.99f, 0.7f, 0.4f, 0.08f));

			AddObject("light", PTObject.Sphere(new Vec3(0f, 3f, 0f), 1.7f, GetMaterial("lightMaterial")));
			AddObject("sphere", PTObject.Sphere(new Vec3(0f, -100.3f, 0f), 100f, GetMaterial("blueMatte")));
			AddObject("cube", PTObject.RectangularPrism(-1f, 1f, 0.2f, 0.45f, -1f, 1f, GetMaterial("glass")));
			AddObject("otherCube", PTObject.RectangularPrism(0f, 0.4f, 0f, 0.4f, 0f, 0.4f, GetMaterial("redMirror"))
				.Rotate((float)Math.PI / 10.4f, PTObject.Alignment.X)
				.Rotate((float)Math.PI / 12.8f, PTObject.Alignment.Z)
				.Translate(new Vec3(-0.4f, 0.75f, -0.4f)));
		}

		private void BtnAddObj_Click(object sender, EventArgs e)
		{
			Form dlg1 = new Form();
			dlg1.ShowDialog();
		}

		private void BtnRender_Click(object sender, EventArgs e)
		{
			pbarDuration.Visible = true;
			Task.Run(() =>
			{
				string time = RenderScene();
				pbarDuration.Visible = false;
				lblRenderTime.Text = time;
			});
		}

		private string RenderScene()
		{
			pboxPreview.Image = null;


			int width = int.Parse(txtbxWidth.Text);
			int height = int.Parse(txtbxHeight.Text);
			int samples = int.Parse(txtbxSamples.Text);
			int chunkSize = int.Parse(txtbxChunkSize.Text);

			string fname = "Test";
			if (txtbxFileName.Text != "")
			{
				fname = txtbxFileName.Text;
			}


			List<PTObject> activeObjects = new List<PTObject>();
			foreach (string objName in listbxObjects.CheckedItems)
			{
				activeObjects.Add(GetObject(objName));
				
			}
			PTObject scene = PTObject.HittableList(activeObjects.ToArray());

			System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
			sw.Start();
			if (!PTObject.RenderSceneChunked(width, height, samples, chunkSize, fname, new Vec3(2f, 1.2f, -2f), new Vec3(-2f, 0f, 2f), new Vec3(0f, 1f, 0f), (float)Math.PI / 3f, (float)width / height, scene))
			{
				// all objects are invalidated if rendering fails.
				foreach (string materialName in materials.Keys)
				{
					GetMaterial(materialName).Destroy();
					listbxMaterials.Items.Remove(materialName);
				}
				foreach (string objectName in objects.Keys)
				{
					GetObject(objectName).Destroy();
					listbxObjects.Items.Remove(objectName);
				}
				Console.WriteLine("\nRendering failed.");
			}
			sw.Stop();

			scene.Destroy(false);

			Bitmap image = ReadBitmapFromPPM(Directory.GetCurrentDirectory() + "\\" + txtbxFileName.Text + ".ppm");
			pboxPreview.Image = image;
			return $"Last Render Took: {sw.ElapsedMilliseconds} ms";
		}

		private static Bitmap ReadBitmapFromPPM(string file)
		{
			var reader = new BinaryReader(new FileStream(file, FileMode.Open));
			if (reader.ReadChar() != 'P' || reader.ReadChar() != '6') return null;
			reader.ReadChar(); // Eat newline
			string widths = "";
			string heights = "";
			char temp;
			while ((temp = reader.ReadChar()) != ' ')
			{
				widths += temp;
			}
			while ((temp = reader.ReadChar()) >= '0' && temp <= '9')
			{
				heights += temp;
			}
			if (reader.ReadChar() != '2' || reader.ReadChar() != '5' || reader.ReadChar() != '5') return null;
			reader.ReadChar(); // Eat the last newline
			int width = int.Parse(widths);
			int height = int.Parse(heights);
			Bitmap bitmap = new Bitmap(width, height);
			// Read in the pixels
			for (int y = 0; y < height; y++)
			{
				for (int x = 0; x < width; x++)
				{
					int red = reader.ReadByte();
					int green = reader.ReadByte();
					int blue = reader.ReadByte();
					Color color = Color.FromArgb(red, green, blue);
					bitmap.SetPixel(x, y, color);
				}
			}
			reader.Close();
			return bitmap;
		}
	}
}
