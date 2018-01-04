## Hydra Renderer

The Hydra Renderer consists of 3 heads:

- End User Plugin (3ds max or else)
- HydraAPI (infrastructure)
- HydraCore (render engine, compute core)

This repo contain the last one.


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
