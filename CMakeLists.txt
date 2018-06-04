cmake_minimum_required(VERSION 3.5)
project(HydraAPP_Ex)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory (shaderpack)
add_subdirectory (vsgl3)
add_subdirectory (hydra_drv)
add_subdirectory (hydra_app)


#add_subdirectory (${CMAKE_SOURCE_DIR}/../HydraAPI)
