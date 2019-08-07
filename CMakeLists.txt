cmake_minimum_required(VERSION 3.5)
project(HydraAPP_Ex)

set(CMAKE_CXX_STANDARD 14)

set(HYDRA_API_ROOT_DIR ${CMAKE_SOURCE_DIR}/../HydraAPI)
set(HYDRA_API_SRC_DIR ${HYDRA_API_ROOT_DIR}/hydra_api)
set(HYDRA_API_LIB_DIR ${HYDRA_API_ROOT_DIR}/bin)
set(HYDRA_API_UTILS_SRC_DIR ${HYDRA_API_ROOT_DIR}/utils)

add_subdirectory (shaderpack)
add_subdirectory (vsgl3)
add_subdirectory (hydra_drv)
add_subdirectory (hydra_app)


#add_subdirectory (${CMAKE_SOURCE_DIR}/../HydraAPI)
