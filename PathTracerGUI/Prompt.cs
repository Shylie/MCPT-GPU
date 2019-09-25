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
			string name = "";
			Form prompt = new Form
			{
				Text = obj.GetType().Name,
				Width = 360
			};
			PropertyInfo[] propertyInfos = obj.GetType().GetProperties().Where(info => info.Name != nameof(PTObject.Valid)).ToArray();
			if (propertyInfos.Length > 0)
			{
				prompt.Height = 70 * propertyInfos.Length + 65;
				for (int i = 0; i < propertyInfos.Length; i++)
				{
					if (propertyInfos[i].PropertyType == typeof(float))
					{
						Label floatlbl = new Label() { Text = propertyInfos[i].Name, Left = 15, Top = 70 * i - 45 };
						TextBox floatbox = new TextBox() { Name = $"{i}", Text = propertyInfos[i].GetValue(obj).ToString(), Width = 90, Left = 15, Top = 70 * i - 20 };
						prompt.Controls.Add(floatlbl);
						prompt.Controls.Add(floatbox);
					}
					else if (propertyInfos[i].PropertyType == typeof(bool))
					{
						Label boollbl = new Label() { Text = propertyInfos[i].Name, Left = 15, Top = 70 * i - 45 };
						TextBox boolbox = new TextBox() { Name = $"{i}", Text = propertyInfos[i].GetValue(obj).ToString(), Width = 90, Left = 15, Top = 70 * i - 20 };
						prompt.Controls.Add(boollbl);
						prompt.Controls.Add(boolbox);
					}
					else if (propertyInfos[i].PropertyType == typeof(Vec3))
					{
						Label xlbl = new Label() { Text = propertyInfos[i].Name, Left = 15, Top = 70 * i - 45 };
						Label ylbl = new Label() { Text = propertyInfos[i].Name, Left = 115, Top = 70 * i - 45 };
						Label zlbl = new Label() { Text = propertyInfos[i].Name, Left = 215, Top = 70 * i - 45 };
						TextBox xbox = new TextBox() { Name = $"{i} x", Text = ((Vec3)propertyInfos[i].GetValue(obj)).X.ToString(), Width = 90, Left = 15, Top = 70 * i - 20 };
						TextBox ybox = new TextBox() { Name = $"{i} y", Text = ((Vec3)propertyInfos[i].GetValue(obj)).Y.ToString(), Width = 90, Left = 115, Top = 70 * i - 20 };
						TextBox zbox = new TextBox() { Name = $"{i} z", Text = ((Vec3)propertyInfos[i].GetValue(obj)).Z.ToString(), Width = 90, Left = 215, Top = 70 * i - 20 };
						prompt.Controls.Add(xlbl);
						prompt.Controls.Add(ylbl);
						prompt.Controls.Add(zlbl);
						prompt.Controls.Add(xbox);
						prompt.Controls.Add(ybox);
						prompt.Controls.Add(zbox);
					}
					else if (propertyInfos[i].PropertyType == typeof(PTObject.Alignment))
					{
						Label alignmentlbl = new Label() { Text = propertyInfos[i].Name, Left = 15, Top = 70 * i - 45 };
						TextBox alignmentbox = new TextBox() { Name = $"{i}", Text = propertyInfos[i].GetValue(obj).ToString(), Width = 90, Left = 15, Top = 70 * i - 20 };
						prompt.Controls.Add(alignmentlbl);
						prompt.Controls.Add(alignmentbox);
					}
					else if (propertyInfos[i].PropertyType.IsSubclassOf(typeof(PTObject)) || propertyInfos[i].PropertyType == typeof(PTObject))
					{
						Label ptlbl = new Label() { Text = propertyInfos[i].Name, Left = 15, Top = 70 * i - 45 };
						string txt = "";
						foreach (string key in ptObjects.Keys)
						{
							if (ReferenceEquals(ptObjects[key], propertyInfos[i].GetValue(obj)))
							{
								txt = key;
								break;
							}
						}
						TextBox ptbox = new TextBox() { Name = $"{i}", Text = txt, Width = 120, Left = 15, Top = 70 * i - 20 };
						prompt.Controls.Add(ptlbl);
						prompt.Controls.Add(ptbox);
					}
				}
				string idtxt = "";
				foreach (string key in ptObjects.Keys)
				{
					if (ReferenceEquals(ptObjects[key], obj))
					{
						idtxt = key;
						break;
					}
				}
				TextBox namebox = new TextBox() { Name = "name input", Width = 120, Left = 15, Top = prompt.Height - 105, Text = idtxt };
				prompt.Controls.Add(namebox);
				Button submit = new Button() { Left = 15, Top = prompt.Height - 70, Text = "Confirm" };
				submit.Click += (sender, args) =>
				{
					for (int i = 0; i < propertyInfos.Length; i++)
					{
						if (propertyInfos[i].PropertyType == typeof(float))
						{
							propertyInfos[i].SetValue(obj, float.Parse(prompt.Controls[$"{i}"].Text));
						}
						else if (propertyInfos[i].PropertyType == typeof(bool))
						{
							propertyInfos[i].SetValue(obj, bool.Parse(prompt.Controls[$"{i}"].Text));
						}
						else if (propertyInfos[i].PropertyType == typeof(Vec3))
						{
							float x = float.Parse(prompt.Controls[$"{i} x"].Text);
							float y = float.Parse(prompt.Controls[$"{i} y"].Text);
							float z = float.Parse(prompt.Controls[$"{i} z"].Text);
							propertyInfos[i].SetValue(obj, new Vec3(x, y, z));
						}
						else if (propertyInfos[i].PropertyType == typeof(PTObject.Alignment))
						{
							switch (prompt.Controls[$"{i}"].Text)
							{
								case "X":
									propertyInfos[i].SetValue(obj, PTObject.Alignment.X);
									break;
								case "Y":
									propertyInfos[i].SetValue(obj, PTObject.Alignment.Y);
									break;
								case "Z":
									propertyInfos[i].SetValue(obj, PTObject.Alignment.Z);
									break;
								case "XY":
									propertyInfos[i].SetValue(obj, PTObject.Alignment.X | PTObject.Alignment.Y);
									break;
								case "XZ":
									propertyInfos[i].SetValue(obj, PTObject.Alignment.X | PTObject.Alignment.Z);
									break;
								case "YZ":
									propertyInfos[i].SetValue(obj, PTObject.Alignment.Y | PTObject.Alignment.Z);
									break;
								case "XYZ":
									propertyInfos[i].SetValue(obj, PTObject.Alignment.X | PTObject.Alignment.Y | PTObject.Alignment.Z);
									break;
								default:
									propertyInfos[i].SetValue(obj, PTObject.Alignment.None);
									break;
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

		public static (string name, T obj) ShowDialog<T>(Dictionary<string, PTObject> ptObjects) where T : PTObject
		{
			T value = null;
			string name = "";
			Form prompt = new Form
			{
				Text = typeof(T).Name,
				Width = 360
			};
			ConstructorInfo constructorInfo = typeof(T).GetConstructors().Where(ci => ci.GetParameters().Length > 0).First();
			if (constructorInfo != null)
			{
				ParameterInfo[] parameterInfos = constructorInfo.GetParameters();
				prompt.Height = 35 * parameterInfos.Length + 115;
				for (int i = 0; i < parameterInfos.Length; i++)
				{
					if (parameterInfos[i].ParameterType == typeof(float))
					{
						TextBox floatbox = new TextBox() { Name = $"{i}", Text = parameterInfos[i].Name, Width = 60, Left = 15, Top = 35 * (i + 1) - 20 };
						prompt.Controls.Add(floatbox);
					}
					else if (parameterInfos[i].ParameterType == typeof(bool))
					{
						TextBox boolbox = new TextBox() { Name = $"{i}", Text = parameterInfos[i].Name, Width = 60, Left = 15, Top = 35 * (i + 1) - 20 };
						prompt.Controls.Add(boolbox);
					}
					else if (parameterInfos[i].ParameterType == typeof(Vec3))
					{
						TextBox xbox = new TextBox() { Name = $"{i} x", Text = parameterInfos[i].Name + " X", Width = 60, Left = 15, Top = 35 * (i + 1) - 20 };
						TextBox ybox = new TextBox() { Name = $"{i} y", Text = parameterInfos[i].Name + " Y", Width = 60, Left = 90, Top = 35 * (i + 1) - 20 };
						TextBox zbox = new TextBox() { Name = $"{i} z", Text = parameterInfos[i].Name + " Z", Width = 60, Left = 165, Top = 35 * (i + 1) - 20 };
						prompt.Controls.Add(xbox);
						prompt.Controls.Add(ybox);
						prompt.Controls.Add(zbox);
					}
					else if (parameterInfos[i].ParameterType == typeof(PTObject.Alignment))
					{
						TextBox alignmentbox = new TextBox() { Name = $"{i}", Text = parameterInfos[i].Name, Width = 60, Left = 15, Top = 35 * (i + 1) - 20 };
						prompt.Controls.Add(alignmentbox);
					}
					else if (parameterInfos[i].ParameterType.IsSubclassOf(typeof(PTObject)) || parameterInfos[i].ParameterType == typeof(PTObject))
					{
						TextBox ptbox = new TextBox() { Name = $"{i}", Text = parameterInfos[i].Name + " ID", Width = 120, Left = 15, Top = 35 * (i + 1) - 20 };
						prompt.Controls.Add(ptbox);
					}
				}
				TextBox namebox = new TextBox() { Name = "name input", Width = 120, Left = 15, Top = prompt.Height - 105, Text = "ID" };
				prompt.Controls.Add(namebox);
				Button submit = new Button() { Left = 15, Top = prompt.Height - 70, Text = "Confirm" };
				submit.Click += (sender, args) =>
				{
					object[] pars = new object[parameterInfos.Length];
					for (int i = 0; i < parameterInfos.Length; i++)
					{
						if (parameterInfos[i].ParameterType == typeof(float))
						{
							pars[i] = float.Parse(prompt.Controls[$"{i}"].Text);
						}
						else if (parameterInfos[i].ParameterType == typeof(bool))
						{
							pars[i] = bool.Parse(prompt.Controls[$"{i}"].Text);
						}
						else if (parameterInfos[i].ParameterType == typeof(Vec3))
						{
							float x = float.Parse(prompt.Controls[$"{i} x"].Text);
							float y = float.Parse(prompt.Controls[$"{i} y"].Text);
							float z = float.Parse(prompt.Controls[$"{i} z"].Text);
							pars[i] = new Vec3(x, y, z);
						}
						else if (parameterInfos[i].ParameterType == typeof(PTObject.Alignment))
						{
							switch (prompt.Controls[$"{i}"].Text)
							{
								case "X":
									pars[i] = PTObject.Alignment.X;
									break;
								case "Y":
									pars[i] = PTObject.Alignment.Y;
									break;
								case "Z":
									pars[i] = PTObject.Alignment.Z;
									break;
								case "XY":
									pars[i] = PTObject.Alignment.X | PTObject.Alignment.Y;
									break;
								case "XZ":
									pars[i] = PTObject.Alignment.X | PTObject.Alignment.Z;
									break;
								case "YZ":
									pars[i] = PTObject.Alignment.Y | PTObject.Alignment.Z;
									break;
								case "XYZ":
									pars[i] = PTObject.Alignment.X | PTObject.Alignment.Y | PTObject.Alignment.Z;
									break;
								default:
									pars[i] = PTObject.Alignment.None;
									break;
							}
						}
						else if (parameterInfos[i].ParameterType.IsSubclassOf(typeof(PTObject)) || parameterInfos[i].ParameterType == typeof(PTObject))
						{
							pars[i] = ptObjects[prompt.Controls[$"{i}"].Text];
						}
					}
					value = constructorInfo.Invoke(pars) as T;
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
			return (name, value);
		}
	}
}
