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
using PathTracerNET.Materials;
using PathTracerNET.Textures;

namespace PathTracerGUI
{
	internal partial class Main : Form
	{
		private readonly Dictionary<string, Texture> textures = new Dictionary<string, Texture>();
		private readonly Dictionary<string, Material> materials = new Dictionary<string, Material>();
		private readonly Dictionary<string, Hittable> hittables = new Dictionary<string, Hittable>();

		public void AddTexture(string name, Texture texture)
		{
			textures.Add(name, texture);
			listbxTextures.Items.Add(name);
		}

		public Texture GetTexture(string name) => textures[name];

		public void AddMaterial(string name, Material material)
		{
			materials.Add(name, material);
			listbxMaterials.Items.Add(name);
		}

		public Material GetMaterial(string name) => materials[name];

		public void AddHittable(string name, Hittable hittable)
		{
			hittables.Add(name, hittable);
			listbxHittables.Items.Add(name);
		}

		public Hittable GetHittable(string name) => hittables[name];

		public Main()
		{
			InitializeComponent();

			AddTexture("brightWhite", new ConstantTexture(3f, 3f, 3f));
			AddTexture("white", new ConstantTexture(1f, 1f, 1f));
			AddTexture("blue", new ConstantTexture(0.4f, 0.7f, 0.9f));
			AddTexture("yellow", new ConstantTexture(0.8f, 0.7f, 0.4f));
			AddTexture("red", new ConstantTexture(0.99f, 0.56f, 0.4f));

			AddMaterial("lightMaterial", new DiffuseLight(GetTexture("brightWhite")));
			AddMaterial("glass", new Dieletric(GetTexture("white"), 1.54f));
			AddMaterial("blueMatte", new Lambertian(GetTexture("blue")));
			AddMaterial("yellowMatte", new Lambertian(GetTexture("yellow")));
			AddMaterial("redMirror", new Metal(GetTexture("red"), 0.08f));

			AddHittable("light", new Sphere() { CenterXExpression = "0", CenterYExpression = "3", CenterZExpression = "0", RadiusExpression = "1.7", Material = GetMaterial("lightMaterial") });
			AddHittable("sphere", new Sphere() { CenterXExpression = "0", CenterYExpression = "-100.3", CenterZExpression = "0", RadiusExpression = "100", Material = GetMaterial("blueMatte") });
		}

		private void BtnAddObj_Click(object sender, EventArgs e)
		{
			Dictionary<string, PTObject> temp = new Dictionary<string, PTObject>();
			foreach (KeyValuePair<string, Texture> pair in textures) temp.Add(pair.Key, pair.Value);
			foreach (KeyValuePair<string, Material> pair in materials) temp.Add(pair.Key, pair.Value);
			foreach (KeyValuePair<string, Hittable> pair in hittables) temp.Add(pair.Key, pair.Value);
			(string name, PTObject obj) val = ("", null);
			switch (ptObjectTypeSelector.GetItemText(ptObjectTypeSelector.SelectedItem))
			{
				case "Sphere":
					val.obj = new Sphere();
					val.name = Prompt.ShowDialog(val.obj, temp);
					break;
				case "Rotation":
					val.obj = new Rotation();
					val.name = Prompt.ShowDialog(val.obj, temp);
					break;
				case "Translation":
					val.obj = new Translation();
					val.name = Prompt.ShowDialog(val.obj, temp);
					break;
				case "RectangularPlane":
					val.obj = new RectangularPlane();
					val.name = Prompt.ShowDialog(val.obj, temp);
					break;
				case "TriangularPlane":
					val.obj = new TriangularPlane();
					val.name = Prompt.ShowDialog(val.obj, temp);
					break;
				case "Dieletric":
					val.obj = new Dieletric();
					val.name = Prompt.ShowDialog(val.obj, temp);
					break;
				case "DiffuseLight":
					val.obj = new DiffuseLight();
					val.name = Prompt.ShowDialog(val.obj, temp);
					break;
				case "Lambertian":
					val.obj = new Lambertian();
					val.name = Prompt.ShowDialog(val.obj, temp);
					break;
				case "Metal":
					val.obj = new Metal();
					val.name = Prompt.ShowDialog(val.obj, temp);
					break;
				case "ConstantTexture":
					val.obj = new ConstantTexture();
					val.name = Prompt.ShowDialog(val.obj, temp);
					break;
				case "CheckerboardTexture":
					val.obj = new CheckerboardTexture();
					val.name = Prompt.ShowDialog(val.obj, temp);
					break;
				case "NoiseTexture":
					val.obj = new NoiseTexture();
					val.name = Prompt.ShowDialog(val.obj, temp);
					break;
			}
			if (val.obj != null && val.name != null &&  !temp.ContainsKey(val.name))
			{
				switch (val.obj.Kind)
				{
					case PTObject.PTObjectKind.Hittable:
						AddHittable(val.name, val.obj as Hittable);
						break;
					case PTObject.PTObjectKind.Material:
						AddMaterial(val.name, val.obj as Material);
						break;
					case PTObject.PTObjectKind.Texture:
						AddTexture(val.name, val.obj as Texture);
						break;
				}
			}
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

		private void ListbxMaterials_MouseUp(object sender, MouseEventArgs e)
		{
			if (e.Button == MouseButtons.Right && sender is CheckedListBox checkedListBox)
			{
				int idx = checkedListBox.IndexFromPoint(e.Location);
				if (idx >= 0)
				{
					checkedListBox.SelectedIndex = idx;
					Dictionary<string, PTObject> temp = new Dictionary<string, PTObject>();
					foreach (KeyValuePair<string, Texture> pair in textures) temp.Add(pair.Key, pair.Value);
					foreach (KeyValuePair<string, Material> pair in materials) temp.Add(pair.Key, pair.Value);
					foreach (KeyValuePair<string, Hittable> pair in hittables) temp.Add(pair.Key, pair.Value);
					string name = Prompt.ShowDialog(GetMaterial(checkedListBox.Items[idx].ToString()), temp);
					if (name == checkedListBox.Items[idx].ToString()) return;
					materials.Add(name, materials[checkedListBox.Items[idx].ToString()]);
					materials.Remove(checkedListBox.Items[idx].ToString());
					checkedListBox.Items[idx] = name;
				}
			}
		}

		private void ListbxHittables_MouseUp(object sender, MouseEventArgs e)
		{
			if (e.Button == MouseButtons.Right && sender is CheckedListBox checkedListBox)
			{
				int idx = checkedListBox.IndexFromPoint(e.Location);
				if (idx >= 0)
				{
					checkedListBox.SelectedIndex = idx;
					Dictionary<string, PTObject> temp = new Dictionary<string, PTObject>();
					foreach (KeyValuePair<string, Texture> pair in textures) temp.Add(pair.Key, pair.Value);
					foreach (KeyValuePair<string, Material> pair in materials) temp.Add(pair.Key, pair.Value);
					foreach (KeyValuePair<string, Hittable> pair in hittables) temp.Add(pair.Key, pair.Value);
					string name = Prompt.ShowDialog(GetHittable(checkedListBox.Items[idx].ToString()), temp);
					if (name == checkedListBox.Items[idx].ToString()) return;
					hittables.Add(name, hittables[checkedListBox.Items[idx].ToString()]);
					hittables.Remove(checkedListBox.Items[idx].ToString());
					checkedListBox.Items[idx] = name;
				}
			}
		}

		private void ListbxTextures_MouseUp(object sender, MouseEventArgs e)
		{
			if (e.Button == MouseButtons.Right && sender is CheckedListBox checkedListBox)
			{
				int idx = checkedListBox.IndexFromPoint(e.Location);
				if (idx >= 0)
				{
					checkedListBox.SelectedIndex = idx;
					Dictionary<string, PTObject> temp = new Dictionary<string, PTObject>();
					foreach (KeyValuePair<string, Texture> pair in textures) temp.Add(pair.Key, pair.Value);
					foreach (KeyValuePair<string, Material> pair in materials) temp.Add(pair.Key, pair.Value);
					foreach (KeyValuePair<string, Hittable> pair in hittables) temp.Add(pair.Key, pair.Value);
					string name = Prompt.ShowDialog(GetTexture(checkedListBox.Items[idx].ToString()), temp);
					if (name == checkedListBox.Items[idx].ToString()) return;
					textures.Add(name, textures[checkedListBox.Items[idx].ToString()]);
					textures.Remove(checkedListBox.Items[idx].ToString());
					checkedListBox.Items[idx] = name;
				}
			}
		}

		private string RenderScene()
		{
			pboxPreview.Image = null;


			int width = int.Parse(txtbxWidth.Text);
			int height = int.Parse(txtbxHeight.Text);
			int samples = int.Parse(txtbxSamples.Text);
			int frames = int.Parse(txtbxFrames.Text);
			double timeStep = double.Parse(txtbxTimeStep.Text);
			int chunkSize = int.Parse(txtbxChunkSize.Text);

			string fname = "Test";
			if (txtbxFileName.Text != "")
			{
				fname = txtbxFileName.Text;
			}

			List<Hittable> activeHittables = new List<Hittable>();
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

			Hittable scene = new HittableList(activeHittables.ToArray());

			Vec3 lookFrom = new Vec3(2f, 1.2f, -2f);
			Vec3 lookAt = new Vec3(-2f, 0f, 2f);
			System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
			sw.Start();
			if (!PTObject.RenderAnimatedSceneChunked(
				width, height, samples, chunkSize, timeStep, frames, fname,
				lookFrom, lookAt, new Vec3(0f, 1f, 0f), (float)Math.PI / 3f, (float)width / height, 0f, (lookFrom - lookAt).Length,
				scene))
			{
				// all objects are invalidated if rendering fails.
				foreach (string textureName in textures.Keys)
				{
					GetTexture(textureName).Destroy();
				}
				textures.Clear();
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
								hittables.Add(i.ToString(), autosaved[i] as Hittable);
								break;
							case PTObject.PTObjectKind.Material:
								materials.Add(i.ToString(), autosaved[i] as Material);
								break;
							case PTObject.PTObjectKind.Texture:
								textures.Add(i.ToString(), autosaved[i] as Texture);
								break;
						}
					}
				}

				return "Rendering failed.";
			}
			sw.Stop();

			scene.Destroy();

			Bitmap image = ReadBitmapFromPPM(Directory.GetCurrentDirectory() + "\\" + fname + "_0.ppm");
			pboxPreview.Image = image;
			return $"Last Render Took: {sw.ElapsedMilliseconds} ms\nAverage: {sw.ElapsedMilliseconds / (double)frames} ms / frame";
		}

		private static Bitmap ReadBitmapFromPPM(string file)
		{
			using (FileStream fs = new FileStream(file, FileMode.Open))
			{
				using (BinaryReader reader = new BinaryReader(fs))
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
}
