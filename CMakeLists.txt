cmake_minimum_required(VERSION 3.9)
project(HydraAPP_Ex)

set(CMAKE_CXX_STANDARD 14)

set(HYDRA_API_ROOT_DIR ${CMAKE_SOURCE_DIR}/../HydraAPI CACHE PATH "Path to HydraAPI")
set(HYDRA_API_SRC_DIR ${HYDRA_API_ROOT_DIR}/hydra_api)
set(HYDRA_API_LIB_DIR ${HYDRA_API_ROOT_DIR}/bin)
set(HYDRA_API_UTILS_SRC_DIR ${HYDRA_API_ROOT_DIR}/utils)

set(ADDITIONAL_INCLUDE_DIRS
        ${HYDRA_API_ROOT_DIR}/dependencies/include CACHE STRING "Path to HydraAPI dependencies (freeimage and, if OpenGL is used, glfw)")

set(USE_GL OFF CACHE BOOL "HydraAPI with OpenGL support")

if(WIN32)
    set(HYDRA_DIR_NAME "[Hydra]")
    set(ADDITIONAL_LIBRARY_DIRS
            ${HYDRA_API_ROOT_DIR}/dependencies/lib_x64_win CACHE STRING "Path to HydraAPI dependencies (freeimage and, if OpenGL is used, glfw)")
else()
    set(HYDRA_DIR_NAME "hydra")
    set(ADDITIONAL_LIBRARY_DIRS
            ${HYDRA_API_ROOT_DIR}/dependencies/lib_x64_linux CACHE STRING "Path to HydraAPI dependencies (freeimage and, if OpenGL is used, glfw)")
endif()

add_subdirectory (shaderpack)
add_subdirectory (vsgl3)
add_subdirectory (hydra_drv)
add_subdirectory (hydra_app)


#add_subdirectory (${CMAKE_SOURCE_DIR}/../HydraAPI)
