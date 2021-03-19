# MCPT-GPU
A basic hardware-accelerated¹ scene rendering program written in C++ and C#. It uses a rendering method called [path tracing](https://en.wikipedia.org/wiki/Path_tracing) and utilizes [CUDA](https://developer.nvidia.com/cuda-toolkit) to speed up the rendering process. The user interface is written in C#; the C++ code that renders the image is called via [P/Invoke](https://docs.microsoft.com/en-us/dotnet/standard/native-interop/pinvoke).

¹Requires an NVIDIA GPU due to use of [CUDA](https://developer.nvidia.com/cuda-toolkit).
# Sample images of the GUI
![](https://i.imgur.com/W4wUX3A.png)
![](https://i.imgur.com/ZUojJ4W.png)
# More sample images
![](https://i.imgur.com/8fGlpHf.png)
