using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;
using System.Reflection;

using PathTracerNET;

namespace PathTracerGUI
{
	internal abstract class Prompt
	{
		public static string ShowDialog<T>(T obj, Dictionary<string, PTObject> ptObjects) where T : PTObject
		{
			string name = null;
			foreach (string key in ptObjects.Keys)
			{
				if (ReferenceEquals(ptObjects[key], obj))
				{
					name = key;
					break;
				}
			}
			Form prompt = new Form
			{
				Text = obj.GetType().Name,
				Width = 360
			};
			PropertyInfo[] propertyInfos = obj.GetType().GetProperties().Where(info => info.Name != nameof(PTObject.Valid)).ToArray();
			if (propertyInfos.Length > 0)
			{
				prompt.Height = 70 * propertyInfos.Length + 115;
				for (int i = 0; i < propertyInfos.Length; i++)
				{
					if (propertyInfos[i].PropertyType == typeof(int) || propertyInfos[i].PropertyType == typeof(float) || propertyInfos[i].PropertyType == typeof(bool) || propertyInfos[i].PropertyType == typeof(string))
					{
						Label lbl = new Label() { Text = propertyInfos[i].Name, Left = 15, Top = 70 * (i + 1) - 45 };
						TextBox tbox = new TextBox() { Name = $"{i}", Text = propertyInfos[i].GetValue(obj)?.ToString() ?? "", Width = 90, Left = 15, Top = 70 * (i + 1) - 20 };
						prompt.Controls.Add(lbl);
						prompt.Controls.Add(tbox);
					}
					else if (propertyInfos[i].PropertyType == typeof(PTObject.Alignment))
					{
						Label alignmentlbl = new Label() { Text = propertyInfos[i].Name, Left = 15, Top = 70 * (i + 1) - 45 };
						TextBox alignmentbox = new TextBox() { Name = $"{i}", Text = propertyInfos[i].GetValue(obj)?.ToString() ?? "", Width = 90, Left = 15, Top = 70 * (i + 1) - 20 };
						prompt.Controls.Add(alignmentlbl);
						prompt.Controls.Add(alignmentbox);
					}
					else if (propertyInfos[i].PropertyType.IsSubclassOf(typeof(Hittable)) || propertyInfos[i].PropertyType == typeof(Hittable))
					{
						Label htlbl = new Label() { Text = propertyInfos[i].Name, Left = 15, Top = 70 * (i + 1) - 45 };
						ComboBox htbox = new ComboBox() { Name = $"{i}", Width = 120, Left = 15, Top = 70 * (i + 1) - 20, DropDownStyle = ComboBoxStyle.DropDownList };
						foreach (string key in ptObjects.Keys)
						{
							if (ptObjects[key] is Hittable hittable && !ReferenceEquals(obj, hittable))
							{
								htbox.Items.Add(key);
								if (ReferenceEquals(hittable, propertyInfos[i].GetValue(obj))) htbox.SelectedItem = key;
							}
						}
						prompt.Controls.Add(htlbl);
						prompt.Controls.Add(htbox);
					}
					else if (propertyInfos[i].PropertyType.IsSubclassOf(typeof(Material)) || propertyInfos[i].PropertyType == typeof(Material))
					{
						Label mtlbl = new Label() { Text = propertyInfos[i].Name, Left = 15, Top = 70 * (i + 1) - 45 };
						ComboBox mtbox = new ComboBox() { Name = $"{i}", Width = 120, Left = 15, Top = 70 * (i + 1) - 20, DropDownStyle = ComboBoxStyle.DropDownList };
						foreach (string key in ptObjects.Keys)
						{
							if (ptObjects[key] is Material material && !ReferenceEquals(obj, material))
							{
								mtbox.Items.Add(key);
								if (ReferenceEquals(material, propertyInfos[i].GetValue(obj))) mtbox.SelectedItem = key;
							}
						}
						prompt.Controls.Add(mtlbl);
						prompt.Controls.Add(mtbox);
					}
					else if (propertyInfos[i].PropertyType.IsSubclassOf(typeof(Texture)) || propertyInfos[i].PropertyType == typeof(Texture))
					{
						Label txlbl = new Label() { Text = propertyInfos[i].Name, Left = 15, Top = 70 * (i + 1) - 45 };
						ComboBox txbox = new ComboBox() { Name = $"{i}", Width = 120, Left = 15, Top = 70 * (i + 1) - 20, DropDownStyle = ComboBoxStyle.DropDownList };
						foreach (string key in ptObjects.Keys)
						{
							if (ptObjects[key] is Texture texture && !ReferenceEquals(obj, texture))
							{
								txbox.Items.Add(key);
								if (ReferenceEquals(texture, propertyInfos[i].GetValue(obj))) txbox.SelectedItem = key;
							}
						}
						prompt.Controls.Add(txlbl);
						prompt.Controls.Add(txbox);
					}
					else if (propertyInfos[i].PropertyType.IsSubclassOf(typeof(PTObject)) || propertyInfos[i].PropertyType == typeof(PTObject))
					{
						Label ptlbl = new Label() { Text = propertyInfos[i].Name, Left = 15, Top = 70 * (i + 1) - 45 };
						string txt = "";
						foreach (string key in ptObjects.Keys)
						{
							if (ReferenceEquals(ptObjects[key], propertyInfos[i].GetValue(obj)))
							{
								txt = key;
								break;
							}
						}
						TextBox ptbox = new TextBox() { Name = $"{i}", Text = txt, Width = 120, Left = 15, Top = 70 * (i + 1) - 20 };
						prompt.Controls.Add(ptlbl);
						prompt.Controls.Add(ptbox);
					}
				}
				TextBox namebox = new TextBox() { Name = "name input", Width = 120, Left = 15, Top = prompt.Height - 105, Text = name };
				prompt.Controls.Add(namebox);
				Button submit = new Button() { Left = 15, Top = prompt.Height - 70, Text = "Confirm" };
				submit.Click += (sender, args) =>
				{
					for (int i = 0; i < propertyInfos.Length; i++)
					{
						if (propertyInfos[i].PropertyType == typeof(int))
						{
							propertyInfos[i].SetValue(obj, int.Parse(prompt.Controls[$"{i}"].Text));
						}
						else if (propertyInfos[i].PropertyType == typeof(float))
						{
							propertyInfos[i].SetValue(obj, float.Parse(prompt.Controls[$"{i}"].Text));
						}
						else if (propertyInfos[i].PropertyType == typeof(bool))
						{
							propertyInfos[i].SetValue(obj, bool.Parse(prompt.Controls[$"{i}"].Text));
						}
						else if (propertyInfos[i].PropertyType == typeof(string))
						{
							propertyInfos[i].SetValue(obj, prompt.Controls[$"{i}"].Text);
						}
						else if (propertyInfos[i].PropertyType == typeof(PTObject.Alignment))
						{
							PTObject.Alignment alignment = PTObject.Alignment.None;
							if (prompt.Controls[$"{i}"].Text.Contains("X")) alignment |= PTObject.Alignment.X;
							if (prompt.Controls[$"{i}"].Text.Contains("Y")) alignment |= PTObject.Alignment.Y;
							if (prompt.Controls[$"{i}"].Text.Contains("Z")) alignment |= PTObject.Alignment.Z;
							propertyInfos[i].SetValue(obj, alignment);
						}
						else if (propertyInfos[i].PropertyType.IsSubclassOf(typeof(Hittable)) || propertyInfos[i].PropertyType == typeof(Hittable))
						{
							if ((prompt.Controls[$"{i}"] as ComboBox)?.SelectedItem is string key && ptObjects[key] is Hittable hittable)
							{
								propertyInfos[i].SetValue(obj, hittable);
							}
						}
						else if (propertyInfos[i].PropertyType.IsSubclassOf(typeof(Material)) || propertyInfos[i].PropertyType == typeof(Material))
						{
							if ((prompt.Controls[$"{i}"] as ComboBox)?.SelectedItem is string key && ptObjects[key] is Material material)
							{
								propertyInfos[i].SetValue(obj, material);
							}
						}
						else if (propertyInfos[i].PropertyType.IsSubclassOf(typeof(Texture)) || propertyInfos[i].PropertyType == typeof(Texture))
						{
							if ((prompt.Controls[$"{i}"] as ComboBox)?.SelectedItem is string key && ptObjects[key] is Texture texture)
							{
								propertyInfos[i].SetValue(obj, texture);
							}
						}
						else if (propertyInfos[i].PropertyType.IsSubclassOf(typeof(PTObject)) || propertyInfos[i].PropertyType == typeof(PTObject))
						{
							propertyInfos[i].SetValue(obj, ptObjects[prompt.Controls[$"{i}"].Text]);
						}
					}
					name = prompt.Controls["name input"].Text;
					prompt.Close();
				};
				prompt.Controls.Add(submit);
				prompt.ShowDialog();
				prompt.Dispose();
			}
			else
			{
				prompt.Dispose();
			}
			return name;
		}
	}
}
