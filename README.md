## Hydra Renderer
The Hydra Renderer consists of 3 heads:

- End User Plugin (3ds max or else)
- HydraAPI (infrastructure)
- HydraCore (render engine, compute core)

This repo contains the last one.

# Build and install

### CMake:
1. Clone HydraAPI repo in some folder (for example "myfolder/HydraAPI"). 
2. Build HydraAPI using provided [instructions](https://github.com/Ray-Tracing-Systems/HydraAPI/blob/master/README.md).
3. Clone HydraCore repo.
4. Set CMake variables:
    - HYDRA_API_ROOT_DIR to path to HydraAPI source dir;
    - USE_GL to the same value as HydraAPI build. On Windows you will need to set USE_GL=ON.
5. Build with CMake. Example command to build and install HydraCore (for example from "myfolder/HydraCore/build"):   
```shell
cmake -DUSE_GL=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/home/YourUserName .. && make all install -j 4
````
Notes:
If shaders are recompiled every time you run Hydra,
check **input.cpp** - variable **inDevelopment** should be set to **false**:
```shell
inDevelopment = false;
```
6. Make sure that OpenCL is installed. This command should display the full information about your graphics card.
```
clinfo
```
If this is not the case, install OpenCL. An example for Linux(Ubuntu):
#### 1.Install NVIDIA proprietary drivers:
```
sudo apt update
sudo ubuntu-drivers autoinstall
sudo reboot
```
#### 2.Check the driver operation:
```
nvidia-smi
```
If you see information about your GPU, the driver is working correctly.

#### 3.Install the necessary OpenCL packages:
```
sudo apt install acl-icq-libopencl1 nvidia-opencl-device
```
#### 4.Check for OpenCL configuration:
```
ls /etc/OpenCL/vendors/
```
There should be an nvidia.icd file.

#### 5.Check the OpenCL installation:
```
clinfo
```
If you see the full information - OpenCL is installed.



## Windows specific

### Windows MSVC:
HydraAPI and HydraCore can be built using provided MSVC solutions as well as CMake.
It can be necessary if you need it to compile 3ds Max plugin.
1. Clone HydraAPI repo is some folder (for example "myfolder/HydraAPI").
2. Build HydraAPI with visual studio 2019 or later.
3. Clone HydraCore repo in the same folder (to form "myfolder/HydraCore").
4. Set **"inDevelopment = false"** inside "input.cpp".
5. NVidia graphics cards have their own shader cache, so it's better to clear it.
6. Build HydraCore with visual studio 2019 or later.

### Windows installation
1. Run shaderpack (set it as a startup project and then press Ctr+F5).
2. Move all ".xx" files from "HydraCore/hydra_drv/shaders" to "C:/[Hydra]/bin2/shaders/".
3. Copy file "texproc.cl" from "HydraCore/hydra_drv/shaders" to "C:/[Hydra]/bin2/shaders/".
4. Copy files "cfetch.h" and "cglobals.h" from "HydraCore/hydra_drv" to "C:/[Hydra]/bin2/shaders/"
5. Copy built hydra.exe to "C:/[Hydra]/bin2"
6. Delete all files inside  "C:/[Hydra]/bin2/shadercache/" if you have them. This will clear Hydra shader cache.

### Windows running
You will need to place dll files (FreeImage.dll, glfw3.dll) together with executable.

# Command line examples

* simple image render
```bash
hydra -inputlib "tests/test_42" -out "/home/frol/temp/z_out.png" -nowindow 1 
```

* simple image render (takes statefile "tests/test_42/statex_00012.xml")
```bash
hydra -inputlib "tests/test_42" -statefile "statex_00012.xml" -out "/home/frol/temp/z_out.png" -nowindow 1 
```

* rendering on 2 GPUs via OS shared memory (via HydraAPI only!)
```bash
hydra -nowindow 1 -inputlib "tests/test_42" -width 1024 -height 768 -cpu_fb 0 -sharedimage hydraimage_1533639330288 -cl_device_id 0
hydra -nowindow 1 -inputlib "tests/test_42" -width 1024 -height 768 -cpu_fb 0 -sharedimage hydraimage_1533639330288 -cl_device_id 1
```

# Building Embree (if you need it for some reason under your custom OS)

Unix:

1. Clone embree2 (we used 2.17 last time). **#NOTE:** do not use embree3, it will not work.
2. install cmake curses (ccmake).
3. mkdir build
4. ccmake ..
5. set EMBREE_MAX_ISA to SSE2 or SSE4.2; **#NOTE:** this is important! Other will not work due to different BVH layout.
6. set EMBREE_TASKING_SYSTEM to INTERNAL
7. set EMBREE_STATIC_LIB to ON
8. build embree. Press 'c', then 'g' and quit from ccmake. Then exec "make -j 4".
   Make sure everything works (for example any of their samples). 
9. copy all files from "HydraCore/bvh_builder" to "embree2/tutorials/bvh_access". Replace all.
10. repeat 8 step, exec "make -j 4" from "embree2/build" folder again.
    Now you should get "libhydrabvhbuilder.a" inside "embree2/build" folder.
11. Copy several files to "HydraCore/LIBRARY/lib_x64_linux" (or configure your own OS folder via CMake):
    libembree.a
    libhydrabvhbuilder.a
    liblexers.a
    libsimd.a
    libsys.a
    libtasking.a

Windows:

It's almost the same except that you need to pack all to the single "bvh_builder.dll" file. \
We usually edit project for "embree2/tutorials/bvh_access" in Visual Studio. \
So, you don't have to replace "embree2/tutorials/bvh_access/CMakeLists.txt" with "HydraCore/bvh_builderCMakeLists.txt".

# License and dependencies

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
So in general there should be no problem to use HydraCore in your open source or commercial projects. 

However, if you find that for some reason you can't use one of these components, please let us know!
Most of these components can be replaced.

# Acknowledgments
This project is supported by RFBR 16-31-60048 "mol_a_dk" and 18-31-20032 "mol_a_ved".
