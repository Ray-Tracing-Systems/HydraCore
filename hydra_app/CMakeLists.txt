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
    mesh_utils.cpp
    mesh_utils.h
    Timer.cpp
    Timer.h
    )

set(ADDITIONAL_INCLUDE_DIRS
        ../LIBRARY/include)

find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(main ${SOURCE_FILES})

include_directories(${ADDITIONAL_INCLUDE_DIRS})

