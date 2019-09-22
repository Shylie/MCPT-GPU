using System.Collections.Generic;
using System.Linq;
using System.Windows.Forms;
using System.Reflection;

using PathTracerNET;

namespace PathTracerGUI
{
	internal abstract class Prompt<T> where T : PTObject
	{
		public static T ShowDialog(Dictionary<string, PTObject> ptObjects)
		{
			T value = null;
			Form prompt = new Form
			{
				Width = 360
			};
			ConstructorInfo constructorInfo = typeof(T).GetConstructors().Where(ci => ci.GetParameters().Length > 0).First();
			if (constructorInfo != null)
			{
				ParameterInfo[] parameterInfos = constructorInfo.GetParameters();
				prompt.Height = 35 * parameterInfos.Length + 85;
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
						TextBox zbox = new TextBox() { Name = $"{i} z", Text = parameterInfos[i].Name + "Z", Width = 60, Left = 165, Top = 35 * (i + 1) - 20 };
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
						TextBox ptbox = new TextBox() { Name = $"{i}", Text = parameterInfos[i].Name + " ID", Width = 60, Left = 15, Top = 35 * (i + 1) - 20 };
						prompt.Controls.Add(ptbox);
					}
				}
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
			return value;
		}
	}
}
