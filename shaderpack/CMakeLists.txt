set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES
        pack.cpp)

set(ADDITIONAL_INCLUDE_DIRS
        ../LIBRARY/include)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    MESSAGE(WARNING "DEBUG IS ON")
    add_definitions(-DDEBUG -D_DEBUG)
endif()

add_definitions(-DHYDRA_DRV_PATH="${CMAKE_SOURCE_DIR}/hydra_drv")
add_executable(shaderpack ${SOURCE_FILES})

target_link_libraries(shaderpack LINK_PUBLIC vsgl3 ${HYDRA_API_CLEW})
if(WIN32)
    #add_definitions(-DUNICODE -D_UNICODE)
    #add_definitions(-DNEED_DIR_CHANGE)
else()
    target_compile_options(shaderpack PRIVATE -fpermissive -Wnarrowing)
    target_link_libraries(shaderpack LINK_PUBLIC OpenCL)
endif()

add_custom_command(TARGET shaderpack POST_BUILD
        COMMAND ${CMAKE_BINARY_DIR}/shaderpack/shaderpack
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "packing shaders...")
