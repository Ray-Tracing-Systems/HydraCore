set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES
        cglobals.h
        cfetch.h
        crandom.h
        clight.h
        cmaterial.h
        ctrace.h
        cbidir.h
        AbstractMaterial.h
        Bitmap.cpp
        bitonic_sort_gpu.cpp
        bitonic_sort_gpu.h
        cl_scan_gpu.cpp
        cl_scan_gpu.h
        CPUBilateralFilter2D.cpp
        CPUExp_bxdf.h
        CPUExp_Integrators_Common.cpp
        CPUExp_Integrators.h
        CPUExp_Integrators_LT.cpp
        CPUExp_Integrators_PT.cpp
        CPUExp_Integrators_PT_Loop.cpp
        CPUExp_Integrators_PT_QMC.cpp
        CPUExp_Integrators_SBDPT.cpp
        CPUExp_Integrators_MMLT.cpp
        CPUExp_Integrators_MMLTDebug.cpp
        CPUExp_IntegratorSSS.cpp
        CPUExp_Integrators_ThreeWay.cpp
        CPUExp_Integrators_TwoWay.cpp
        CPUExpLayer.cpp
        FastList.h
        globals_sys.cpp
        globals_sys.h
        GPUOCLData.cpp
        GPUOCLKernels.cpp
        GPUOCLLayer.h
        GPUOCLLayer.cpp
        GPUOCLLayerAdvanced.cpp
        GPUOCLLayerCore.cpp
        GPUOCLLayerOther.cpp
        GPUOCLLayerMLT.cpp
        GPUOCLTests.cpp
        IBVHBuilderAPI.h
        IESRender.cpp
        IHWLayerDataAssembler.cpp
        IHWLayer.h
        IMemoryStorage.h
        MemoryStorageCPU.cpp
        MemoryStorageCPU.h
        MemoryStorageOCL.cpp
        MemoryStorageOCL.h
        PlainLightConverter.cpp
        PlainMaterialConverter.cpp
        qmc_sobol_niederreiter.cpp
        RenderDriverRTE_AlphaTestTable.cpp
        RenderDriverRTE_AuxTextures.cpp
        RenderDriverRTE.cpp
        RenderDriverRTE_DebugBVH.cpp
        RenderDriverRTE.h
        RenderDriverRTE_PdfTables.cpp
        RenderDriverRTE_ProcTex.cpp
        CPUExp_GBuffer.cpp
        ../bakeBrdfEnergy/MSTablesGGX2017.cpp
        ../bakeBrdfEnergy/MSTablesTransp.cpp
        ../cam_plug/SimpleCamPlugin.cpp
    )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
find_package(OpenMP)

if(WIN32)
    set(SOURCE_FILES ${SOURCE_FILES} BVHBuilderLoaderWin.cpp ../cam_plug/HydraDLibWin64.cpp)
else()
    set(SOURCE_FILES ${SOURCE_FILES} ../cam_plug/HydraDLibUnix.cpp)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    #MESSAGE(WARNING "DEBUG IS ON")
    add_definitions(-DDEBUG -D_DEBUG)
endif()

include_directories(${ADDITIONAL_INCLUDE_DIRS})
include_directories(${HYDRA_API_ROOT_DIR})
include_directories(${HYDRA_API_UTILS_SRC_DIR})

ADD_LIBRARY(hydra_drv STATIC ${SOURCE_FILES})

