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

        private Dictionary<String, PTObject> materials = new Dictionary<String, PTObject>();
        private Dictionary<String, PTObject> objects = new Dictionary<String, PTObject>();

        public void addMaterial(String name, PTObject material)
        {
            this.materials.Add(name, material);
            listbxMaterials.Items.Add(name);
        }

        public PTObject getMaterial(String name)
        {
            return this.materials[name];
        }

        public void addObject(String name, PTObject obj)
        {
            this.objects.Add(name, obj);
            listbxObjects.Items.Add(name);
        }

        public PTObject getObject(String name)
        {
            return this.objects[name];
        }

        public Main()
        {
            InitializeComponent();

            addMaterial("lightMaterial", PTObject.DiffuseLight(3f, 3f, 3f));
            addMaterial("glass", PTObject.Dieletric(1f, 1f, 1f, 1.2f));
            addMaterial("blueMatte", PTObject.Lambertian(0.4f, 0.7f, 0.9f));
            addMaterial("yellowMatte", PTObject.Lambertian(0.8f, 0.7f, 0.4f));
            addMaterial("redMirror", PTObject.Metal(0.99f, 0.7f, 0.4f, 0.08f));

            addObject("light", PTObject.Sphere(new Vec3(0f, 3f, 0f), 1.7f, getMaterial("lightMaterial")));
            addObject("sphere", PTObject.Sphere(new Vec3(0f, -100.3f, 0f), 100f, getMaterial("blueMatte")));
            addObject("cube", PTObject.RectangularPrism(-1f, 1f, 0.2f, 0.45f, -1f, 1f, getMaterial("glass")));
            addObject("otherCube", PTObject.RectangularPrism(0f, 0.4f, 0f, 0.4f, 0f, 0.4f, getMaterial("redMirror")));
        }

        private void BtnAddObj_Click(object sender, EventArgs e)
        {
            Form dlg1 = new Form();
            dlg1.ShowDialog();
        }

        private void BtnRender_Click(object sender, EventArgs e)
        {
            pbarDuration.Visible = true;
            String time = RenderScene();
            //multithreading not working currently
            //var t = Task.Run(() => time = RenderScene()); 
            //t.Wait();
            pbarDuration.Visible = false;
            lblRenderTime.Text = time;
        }

        private String RenderScene()
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


            List<PTObject> objects = new List<PTObject>();
            foreach (string objName in listbxObjects.CheckedItems)
            {
                objects.Add(getObject(objName));
                
            }
            PTObject scene = PTObject.HittableList(objects.ToArray());

            System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
            sw.Start();
            if (!PTObject.RenderSceneChunked(width, height, samples, chunkSize, fname, new Vec3(2f, 1.2f, -2f), new Vec3(-2f, 0f, 2f), new Vec3(0f, 1f, 0f), (float)Math.PI / 3f, (float)width / height, scene))
            {
                Console.WriteLine("\nRendering failed.");
            }
            sw.Stop();

            scene.Destroy(false);

            Bitmap image = ReadBitmapFromPPM(System.IO.Directory.GetCurrentDirectory() + "\\" + txtbxFileName.Text + ".ppm");
            pboxPreview.Image = image;
            return String.Format("Last Render Took: {0} ms", sw.ElapsedMilliseconds);
        }

        private static Bitmap ReadBitmapFromPPM(string file)
        {
            var reader = new BinaryReader(new FileStream(file, FileMode.Open));
            if (reader.ReadChar() != 'P' || reader.ReadChar() != '6')
                return null;
            reader.ReadChar(); //Eat newline
            string widths = "", heights = "";
            char temp;
            while ((temp = reader.ReadChar()) != ' ')
                widths += temp;
            while ((temp = reader.ReadChar()) >= '0' && temp <= '9')
                heights += temp;
            if (reader.ReadChar() != '2' || reader.ReadChar() != '5' || reader.ReadChar() != '5')
                return null;
            reader.ReadChar(); //Eat the last newline
            int width = int.Parse(widths),
                height = int.Parse(heights);
            Bitmap bitmap = new Bitmap(width, height);
            //Read in the pixels
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
