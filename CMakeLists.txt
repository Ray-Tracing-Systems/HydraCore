cmake_minimum_required(VERSION 3.10)
project(HydraAPP_Ex)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

set(HYDRA_API_ROOT_DIR ${CMAKE_SOURCE_DIR}/../HydraAPI CACHE PATH "Path to HydraAPI")
set(HYDRA_API_SRC_DIR ${HYDRA_API_ROOT_DIR}/hydra_api)
set(HYDRA_API_LIB_DIR ${HYDRA_API_ROOT_DIR}/bin)
set(HYDRA_API_UTILS_SRC_DIR ${HYDRA_API_ROOT_DIR}/utils)

set(ADDITIONAL_INCLUDE_DIRS
        ${HYDRA_API_ROOT_DIR}/dependencies/include CACHE STRING "Path to HydraAPI dependencies (freeimage and, if OpenGL is used, glfw)")

set(USE_GL OFF CACHE BOOL "HydraAPI with OpenGL support")

if(WIN32)
    find_library(HYDRA_API_CLEW clew HINTS ${HYDRA_API_LIB_DIR} NO_DEFAULT_PATH)
    set(HYDRA_DIR_NAME "[Hydra]")
    set(ADDITIONAL_LIBRARY_DIRS
            ${HYDRA_API_ROOT_DIR}/dependencies/lib_x64_win CACHE STRING "Path to HydraAPI dependencies (freeimage and, if OpenGL is used, glfw)")
else()
    set(HYDRA_DIR_NAME "hydra")
    set(ADDITIONAL_LIBRARY_DIRS
            ${HYDRA_API_ROOT_DIR}/dependencies/lib_x64_linux CACHE STRING "Path to HydraAPI dependencies (freeimage and, if OpenGL is used, glfw)")
endif()

set(HYDRA_INSTALL_BIN2 "${CMAKE_INSTALL_PREFIX}/${HYDRA_DIR_NAME}/bin2/")
add_compile_definitions(HYDRA_BIN2=${HYDRA_INSTALL_BIN2})
add_compile_definitions(HYDRA_CMAKE)

add_subdirectory (shaderpack)
add_subdirectory (vsgl3)
add_subdirectory (hydra_drv)
add_subdirectory (hydra_app)


if(WIN32)
    set_property(DIRECTORY ${CMAKE_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT hydra)
endif()

#add_subdirectory (${CMAKE_SOURCE_DIR}/../HydraAPI)
