set(CMAKE_CXX_STANDARD 17)

set(SOURCE_FILES
        pack.cpp
        )

set(ADDITIONAL_INCLUDE_DIRS
        ../LIBRARY/include)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    MESSAGE(WARNING "DEBUG IS ON")
    add_definitions(-DDEBUG -D_DEBUG)
endif()

link_directories(${CMAKE_SOURCE_DIR}/../HydraAPI/bin/ ${CMAKE_SOURCE_DIR}/LIBRARY/lib_x64_linux/)

add_definitions(-DHYDRA_DRV_PATH="${CMAKE_SOURCE_DIR}/hydra_drv")
add_executable(shaderpack ${SOURCE_FILES})

#MESSAGE(WARNING "CMAKE_SOURCE_DIR: " ${CMAKE_SOURCE_DIR} )

SET (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -ldl")

if(WIN32)
    add_definitions(-DUNICODE -D_UNICODE)
    add_definitions(-DNEED_DIR_CHANGE)
else()
    target_compile_options(shaderpack PRIVATE -fpermissive -Wnarrowing)
endif()

target_link_libraries(shaderpack LINK_PUBLIC
        vsgl3
        OpenCL)

include_directories(${ADDITIONAL_INCLUDE_DIRS})

add_custom_command(TARGET shaderpack POST_BUILD
        COMMAND ${CMAKE_BINARY_DIR}/shaderpack/shaderpack
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "packing shaders..."
        )
