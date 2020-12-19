@echo on

set PATH_HYDRA_BIN2=C:\[Hydra]\bin2
set PATH_HYDRA_CORE=D:\Works\Ray-Tracing_Systems\HydraDevelop\HydraCore


del /Q %PATH_HYDRA_BIN2%\shaders\
del /Q %PATH_HYDRA_BIN2%\shadercache\


copy "%PATH_HYDRA_CORE%\hydra_app\bvh_builder.dll"                "%PATH_HYDRA_BIN2%\"       
copy "%PATH_HYDRA_CORE%\hydra_app\x64\Release\hydra.exe"          "%PATH_HYDRA_BIN2%\"           
copy "%PATH_HYDRA_CORE%\hydra_drv\cfetch.h"                   	  "%PATH_HYDRA_BIN2%\shaders\"  
copy "%PATH_HYDRA_CORE%\hydra_drv\cglobals.h"                     "%PATH_HYDRA_BIN2%\shaders\"  
copy "%PATH_HYDRA_CORE%\hydra_drv\shaders\image.xx"               "%PATH_HYDRA_BIN2%\shaders\"  
copy "%PATH_HYDRA_CORE%\hydra_drv\shaders\light.xx"               "%PATH_HYDRA_BIN2%\shaders\"  
copy "%PATH_HYDRA_CORE%\hydra_drv\shaders\material.xx"            "%PATH_HYDRA_BIN2%\shaders\"  
copy "%PATH_HYDRA_CORE%\hydra_drv\shaders\mlt.xx"                 "%PATH_HYDRA_BIN2%\shaders\"  
copy "%PATH_HYDRA_CORE%\hydra_drv\shaders\screen.xx"              "%PATH_HYDRA_BIN2%\shaders\"  
copy "%PATH_HYDRA_CORE%\hydra_drv\shaders\sort.xx"                "%PATH_HYDRA_BIN2%\shaders\"  
copy "%PATH_HYDRA_CORE%\hydra_drv\shaders\trace.xx"               "%PATH_HYDRA_BIN2%\shaders\"  
copy "%PATH_HYDRA_CORE%\hydra_drv\shaders\texproc.cl"             "%PATH_HYDRA_BIN2%\shaders\"  



pause