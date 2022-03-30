cmake_minimum_required(VERSION 3.5)
project(HydraAPP_Ex)

set(CMAKE_CXX_STANDARD 17)

set(HYDRA_API_ROOT_DIR ${CMAKE_SOURCE_DIR}/../HydraAPI CACHE STRING "HydraAPI source directory")
set(HYDRA_API_SRC_DIR ${HYDRA_API_ROOT_DIR}/hydra_api)
set(HYDRA_API_LIB_DIR ${HYDRA_API_ROOT_DIR}/bin)
set(HYDRA_API_UTILS_SRC_DIR ${HYDRA_API_ROOT_DIR}/utils)

SET(USE_GL OFF CACHE BOOL "Was HydraAPI built with OpenGL render drivers?")
set(ADDITIONAL_INCLUDE_DIRS
        ${HYDRA_API_ROOT_DIR}/dependencies/include)
if(WIN32)
    set(ADDITIONAL_LIBRARY_DIRS
            ${HYDRA_API_ROOT_DIR}/dependencies/lib_x64_win)
else()
    SET(USE_FIND_PACKAGE OFF CACHE BOOL "Use find_package to find system installation of dependencies")
    set(ADDITIONAL_LIBRARY_DIRS
            ${HYDRA_API_ROOT_DIR}/dependencies/lib_x64_linux)
endif()

add_subdirectory (shaderpack)
add_subdirectory (vsgl3)
add_subdirectory (hydra_drv)
add_subdirectory (hydra_app)


#add_subdirectory (${CMAKE_SOURCE_DIR}/../HydraAPI)
