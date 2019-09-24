using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Xml;
using System.Xml.Serialization;

using PathTracerNET;
using PathTracerNET.Hittables;
using PathTracerNET.Hittables.Geometric;
using PathTracerNET.Hittables.Modifier;
using PathTracerNET.Hittables.Plane;
using PathTracerNET.Hittables.SDF;
using PathTracerNET.Materials;

namespace PathTracerGUI
{
	internal partial class Main : Form
	{
		private Dictionary<string, PTObject> materials = new Dictionary<string, PTObject>();
		private Dictionary<string, PTObject> hittables = new Dictionary<string, PTObject>();

		public void AddMaterial(string name, PTObject material)
		{
			materials.Add(name, material);
			listbxMaterials.Items.Add(name);
		}

		public PTObject GetMaterial(string name)
		{
			return materials[name];
		}

		public void AddHittable(string name, PTObject hittable)
		{
			hittables.Add(name, hittable);
			listbxHittables.Items.Add(name);
		}

		public PTObject GetHittable(string name)
		{
			return hittables[name];
		}

		public Main()
		{
			InitializeComponent();

			AddMaterial("lightMaterial", new DiffuseLight(3f, 3f, 3f));
			AddMaterial("glass", new Dieletric(1f, 1f, 1f, 1.54f));
			AddMaterial("blueMatte", new Lambertian(0.4f, 0.7f, 0.9f));
			AddMaterial("yellowMatte", new Lambertian(0.8f, 0.7f, 0.4f));
			AddMaterial("redMirror", new Metal(0.99f, 0.7f, 0.4f, 0.08f));

			AddHittable("light", new Sphere(new Vec3(0f, 3f, 0f), 1.7f, GetMaterial("lightMaterial")));
			AddHittable("sphere", new Sphere(new Vec3(0f, -100.3f, 0f), 100f, GetMaterial("blueMatte")));
			AddHittable("glassBall", new Sphere(new Vec3(1f, 0.4f, 0f), 0.2f, GetMaterial("glass")));
			AddHittable("plane", new RectangularPlane(-2f, -0.4f, 0.1f, 0.9f, 0f, PTObject.Alignment.X | PTObject.Alignment.Y, true, false, GetMaterial("yellowMatte"))
				.Rotate((float)Math.PI / 10.4f, PTObject.Alignment.X)
				.Rotate((float)Math.PI / 12.8f, PTObject.Alignment.Z)
				.Translate(new Vec3(-0.2f, 0f, -0.2f)));
		}

		private void BtnAddObj_Click(object sender, EventArgs e)
		{
			Dictionary<string, PTObject> temp = new Dictionary<string, PTObject>();
			foreach (KeyValuePair<string, PTObject> pair in materials) temp.Add(pair.Key, pair.Value);
			foreach (KeyValuePair<string, PTObject> pair in hittables) temp.Add(pair.Key, pair.Value);
			(string name, PTObject obj) val = ("", null);
			switch (ptObjectTypeSelector.GetItemText(ptObjectTypeSelector.SelectedItem))
			{
				case "Sphere":
					val = Prompt<Sphere>.ShowDialog(temp);
					break;
				case "Rotation":
					val = Prompt<Rotation>.ShowDialog(temp);
					break;
				case "Translation":
					val = Prompt<Translation>.ShowDialog(temp);
					break;
				case "RectangularPlane":
					val = Prompt<RectangularPlane>.ShowDialog(temp);
					break;
				case "TriangularPlane":
					val = Prompt<TriangularPlane>.ShowDialog(temp);
					break;
				case "DistortedSphere":
					val = Prompt<DistortedSphere>.ShowDialog(temp);
					break;
				case "Dieletric":
					val = Prompt<Dieletric>.ShowDialog(temp);
					break;
				case "DiffuseLight":
					val = Prompt<DiffuseLight>.ShowDialog(temp);
					break;
				case "Lambertian":
					val = Prompt<Lambertian>.ShowDialog(temp);
					break;
				case "Metal":
					val = Prompt<Metal>.ShowDialog(temp);
					break;
			}
			if (val.obj != null && !temp.ContainsKey(val.name))
			{
				switch (val.obj.Kind)
				{
					case PTObject.PTObjectKind.Hittable:
						AddHittable(val.name, val.obj);
						break;
					case PTObject.PTObjectKind.Material:
						AddMaterial(val.name, val.obj);
						break;
				}
			}
			temp.Clear();
		}

		private async void BtnRender_Click(object sender, EventArgs e)
		{
			pbarDuration.Visible = true;
			lblRenderTime.Text = await Task.Run(() => RenderScene());
			if (lblRenderTime.Text == "Rendering failed.")
			{
				listbxHittables.Items.Clear();
				listbxMaterials.Items.Clear();
				foreach (string hittableName in hittables.Keys) listbxHittables.Items.Add(hittableName);
				foreach (string materialName in materials.Keys) listbxMaterials.Items.Add(materialName);
			}
			pbarDuration.Visible = false;
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

			List<PTObject> activeHittables = new List<PTObject>();
			foreach (string hittableName in listbxHittables.CheckedItems)
			{
				activeHittables.Add(GetHittable(hittableName));
			}

			XmlSerializer serializer = new XmlSerializer(typeof(PTObject[]));
			
			using (XmlWriter writer = XmlWriter.Create("autosave.xml"))
			{
				List<PTObject> temp = new List<PTObject>();
				foreach (PTObject material in materials.Values)
				{
					temp.Add(material);
				}
				foreach (PTObject hittable in hittables.Values)
				{
					temp.Add(hittable);
				}
				serializer.Serialize(writer, temp.ToArray());
			}

			PTObject scene = new HittableList(activeHittables.ToArray());

			Vec3 lookFrom = new Vec3(2f, 1.2f, -2f);
			Vec3 lookAt = new Vec3(-2f, 0f, 2f);
			System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
			sw.Start();
			if (!PTObject.RenderSceneChunked(
				width, height, samples, chunkSize, fname,
				lookFrom, lookAt, new Vec3(0f, 1f, 0f), (float)Math.PI / 3f, (float)width / height, 0f, (lookFrom - lookAt).Length,
				scene))
			{
				// all objects are invalidated if rendering fails.
				foreach (string materialName in materials.Keys)
				{
					GetMaterial(materialName).Destroy();
				}
				materials.Clear();
				foreach (string hittableName in hittables.Keys)
				{
					GetHittable(hittableName).Destroy();
				}
				hittables.Clear();
				scene.Destroy();

				using (XmlReader reader = XmlReader.Create("autosave.xml"))
				{
					PTObject[] autosaved = serializer.Deserialize(reader) as PTObject[];
					for (int i = 0; i < autosaved.Length; i++)
					{
						switch (autosaved[i].Kind)
						{
							case PTObject.PTObjectKind.Hittable:
								hittables.Add(i.ToString(), autosaved[i]);
								break;
							case PTObject.PTObjectKind.Material:
								materials.Add(i.ToString(), autosaved[i]);
								break;
						}
					}
				}

				return "Rendering failed.";
			}
			sw.Stop();

			scene.Destroy();

			Bitmap image = ReadBitmapFromPPM(Directory.GetCurrentDirectory() + "\\" + fname + ".ppm");
			pboxPreview.Image = image;
			return $"Last Render Took: {sw.ElapsedMilliseconds} ms";
		}

		private static Bitmap ReadBitmapFromPPM(string file)
		{
			using (BinaryReader reader = new BinaryReader(new FileStream(file, FileMode.Open)))
			{
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
				return bitmap;
			}
		}
	}
}
