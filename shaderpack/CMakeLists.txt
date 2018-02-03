set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES
        pack.cpp
        )

set(ADDITIONAL_INCLUDE_DIRS
        ../LIBRARY/include)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    MESSAGE(WARNING "DEBUG IS ON")
    add_definitions(-DDEBUG -D_DEBUG)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
include_directories(${FREEIMAGE_INCLUDE_DIRS})

find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# include_directories(${ADDITIONAL_INCLUDE_DIRS})
# target_include_directories(main PUBLIC ${OPENGL_INCLUDE_DIR})

link_directories(${CMAKE_SOURCE_DIR}/../HydraAPI/bin/ ${CMAKE_SOURCE_DIR}/LIBRARY/lib_x64_linux/)

add_definitions(-DHYDRA_DRV_PATH="${CMAKE_SOURCE_DIR}/hydra_drv")
add_executable(shaderpack ${SOURCE_FILES})


MESSAGE(WARNING "CMAKE_SOURCE_DIR: " ${CMAKE_SOURCE_DIR} )

SET (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -ldl")

if(WIN32)
    #add_definitions(-DUNICODE -D_UNICODE)
    #add_definitions(-DNEED_DIR_CHANGE)
    #target_link_libraries(main LINK_PUBLIC ${OPENGL_gl_LIBRARY} hydra_api glfw3dll )
else()
    target_compile_options(shaderpack PRIVATE -fpermissive -Wnarrowing)
    target_link_libraries(shaderpack LINK_PUBLIC
            hydra_drv
            vsgl3
            hydra_api
            OpenCL)
endif()

include_directories(${ADDITIONAL_INCLUDE_DIRS})

