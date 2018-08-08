set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES
        Camera.h
        input.cpp
        input.h
        main_app_console.cpp
        main_app_tests.cpp
        main_app_window.cpp
        main.cpp
        main.h
        ${CMAKE_SOURCE_DIR}/../HydraAPI/main/Timer.h
        ${CMAKE_SOURCE_DIR}/../HydraAPI/main/Timer.cpp
    )

set(ADDITIONAL_INCLUDE_DIRS
        ../LIBRARY/include)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    MESSAGE(WARNING "DEBUG IS ON")
    add_definitions(-DDEBUG -D_DEBUG)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
find_package(FreeImage REQUIRED)
include_directories(${FREEIMAGE_INCLUDE_DIRS})

find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)


# include_directories(${ADDITIONAL_INCLUDE_DIRS})
# target_include_directories(main PUBLIC ${OPENGL_INCLUDE_DIR})

link_directories(${CMAKE_SOURCE_DIR}/../HydraAPI/bin/ ${CMAKE_SOURCE_DIR}/LIBRARY/lib_x64_linux/)

add_executable(hydra ${SOURCE_FILES})

#MESSAGE(WARNING "CMAKE_SOURCE_DIR: " ${CMAKE_SOURCE_DIR} )

SET (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -ldl")

if(WIN32)
    #add_definitions(-DUNICODE -D_UNICODE)
    #add_definitions(-DNEED_DIR_CHANGE)
    #target_link_libraries(main LINK_PUBLIC ${OPENGL_gl_LIBRARY} hydra_api glfw3dll )
else()
    target_compile_options(hydra PRIVATE -fpermissive -Wnarrowing ${OpenMP_CXX_FLAGS})
    target_link_libraries(hydra LINK_PUBLIC
            hydra_drv
            hydrabvhbuilder embree sys tasking simd lexers
            vsgl3
            hydra_api
            ies_parser
            Threads::Threads
            dl rt stdc++fs
            glfw
            ${OPENGL_gl_LIBRARY}
            ${OPENGL_glu_LIBRARY}
            ${LIBS}
            ${FREEIMAGE_LIBRARIES}
            ${OpenMP_CXX_FLAGS}
            OpenCL)
endif()

include_directories(${ADDITIONAL_INCLUDE_DIRS})


install(TARGETS hydra DESTINATION hydra)

install(FILES ${CMAKE_SOURCE_DIR}/hydra_drv/cfetch.h
              ${CMAKE_SOURCE_DIR}/hydra_drv/cglobals.h
              ${CMAKE_SOURCE_DIR}/hydra_drv/shaders/texproc.cl
              ${CMAKE_SOURCE_DIR}/hydra_drv/shaders/image.xx
              ${CMAKE_SOURCE_DIR}/hydra_drv/shaders/light.xx
              ${CMAKE_SOURCE_DIR}/hydra_drv/shaders/material.xx
              ${CMAKE_SOURCE_DIR}/hydra_drv/shaders/mlt.xx
              ${CMAKE_SOURCE_DIR}/hydra_drv/shaders/screen.xx
              ${CMAKE_SOURCE_DIR}/hydra_drv/shaders/sort.xx
              ${CMAKE_SOURCE_DIR}/hydra_drv/shaders/trace.xx
        DESTINATION hydra/shaders)

install(CODE "FILE(MAKE_DIRECTORY \${ENV}\${CMAKE_INSTALL_PREFIX}/hydra/shadercache)")