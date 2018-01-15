## Hydra Renderer

The Hydra Renderer consists of 3 heads:

- End User Plugin (3ds max or else)
- HydraAPI (infrastructure)
- HydraCore (render engine, compute core)

This repo contain the last one.

# Usage

1. Clone HydraAPI repo is some folder (for example 'myfolder/HydraAPI'). 
2. Build HydraAPI with visual studio 2015 or later under windows.
3. Clone HydraCore repo in the same folder (to form 'myfolder/HydraCore').
4. Build HydraCore with visual studio 2015 or later under windows.
5. Set 'inLibraryPath = "tests/test_42";' inside input.cpp
6. Set 'inDevelopment = true' inside 'GPUOCLLayer.cpp' to disable internal shader cache (note that nvidia use its' own shader cache!!!).
7. If you want to fly around scene set 'm_screen.m_cpuFrameBuffer = false;' to enable gpu frame buffer.
8. Press 'P' for Path Tracing, 'L' for light tracing.

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
