## Hydra Renderer

The Hydra Renderer consists of 3 heads:

- End User Plugin (3ds max or else)
- HydraAPI (infrastructure)
- HydraCore (render engine, compute core)

This repo contain the last one.

# Usage

Windows:
1. Clone HydraAPI repo is some folder (for example 'myfolder/HydraAPI'). 
2. Build HydraAPI with visual studio 2015 or later under windows.
3. Clone HydraCore repo in the same folder (to form 'myfolder/HydraCore').
4. Build HydraCore with visual studio 2015 or later under windows.
5. Set 'inLibraryPath = "tests/test_42";' inside input.cpp; 
6. Set 'inDeviceId = 0' (or other, depends on your OpenCL HW); inside 'input.cpp'; 
7. Set 'inDevelopment = true' inside 'input.cpp' to disable internal shader cache (note that nvidia use its' own shader cache!). 
8. Press 'P' for Path Tracing, 'L' for light tracing and 'B' for IBPT.
9. Render will save images to 'C:/[Hydra]/rendered_images' each 60 seconds if you set "-saveinterval 60" via command line.

Linux:
1. Clone HydraAPI repo is some folder (for example 'myfolder/HydraAPI'). 
2. Build HydraAPI with Cmake.
3. Clone HydraCore repo in the same folder (to form 'myfolder/HydraCore').
4. Build HydraCore with Cmake.
5. Set 'inLibraryPath = "tests/test_42";' inside input.cpp; 
6. Set 'inDeviceId = 0' (or other, depends on your OpenCL HW); inside 'input.cpp'; 
7. Set 'inDevelopment = true' inside 'input.cpp' to disable internal shader cache (note that nvidia use its' own shader cache!). 
8. Press 'P' for Path Tracing, 'L' for light tracing and 'B' for IBPT.

General:
1. Set 'inDeviceId = -1' to run CPU experimental implementation.
2. Set explicitly integrator that you want inside 'IHWLayerDataAssembler.cpp -> CPUSharedData::PrepareEngineGlobals()'

# Install

Windows:
1. Build HydraCore as described above; 
2. Make shure it works - i. e. you see cornell box with teapot;
3. Set 'inDevelopment = false' inside 'input.cpp';
4. Build main project (main_app.exe) and shader packer (shaderpack.exe).
5. Run shader packer from Visual Studio (working directory should be one of HydraCore subdirs, see 'inputfolder = "../hydra_drv').
6. Copy all crypted '.xx' shaders from 'hydra_drv/shaders' to 'C:/[Hydra]/bin2/shaders'.
7. Rename/move 'hydra_app/x64/Release/main_app.exe' to 'C:/[Hydra]/bin2/hydra.exe'
8. Clear all '.bin' files inside 'C:/[Hydra]/bin2/shadercache'
9. Be sure that 'C:/[Hydra]' contatin further subdirectories:
  9.1. 'C:/[Hydra]/bin2/shaders'
  9.2. 'C:/[Hydra]/bin2/shadercache'
  9.3. 'C:/[Hydra]/logs'
  9.4. 'C:/[Hydra]/pluginFiles' (if you are going to use 3ds Max plugin)
  9.5. 'C:/[Hydra]/rendered_images'
  
Linux:
1. Build HydraCore as described above; 
2. Make shure it works - i. e. you see cornell box with teapot;
3. Set 'inDevelopment = false' inside 'input.cpp';
4. Build main project (main) and shader packer (shaderpack).
5. Run shader packer (working directory is not esssential, Cmake will set HYDRA_DRV_PATH).
6. Copy all crypted '.xx' shaders from 'hydra_drv/shaders' to '/home/usename/hydra/shaders'.
7. Rename/move your 'main' (like 'build/hydra_app/main') to '/home/usename/hydra/hydra'
8. Clear all '.bin' files inside '/home/usename/hydra/shadercache'

# Licence and dependency

HydraCore uses MIT licence itself, however it depends on the other software as follows (see doc/licence directory):

* 02 - FreeImage Public License - Version 1.0 (FreeImage is used in the form of binaries)
* 03 - Embree Apache License 2.0 (Embree is used in the form of binaries)
* 04 - xxhash BSD 3-clause "New" or "Revised" (xxhash is used in the form of sources)
* 05 - pugixml MIT licence (pugixml is used in the form of sources)
* 06 - clew Boost Software License - Version 1.0 - August 17th, 2003 (clew is used in the form of sources)
* 07 - IESNA MIT-like licence (IESNA used in the form of sources)
* 08 - glad MIT licence (glad is used in form of generated source code).
* 09 - glfw BSD-like license (glfw is used in form of binaries only for demonstration purposes).

Most of them are simple MIT-like-licences without any serious restrictions. 
So in general there should be no problem to use HydraCore in your open source or commertial projects. 

However if you find that for some reason you can't use one of these components, please let us know!
Most of these components can be replaced.
