cmake_minimum_required(VERSION 3.14)
project(nnue_network)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# Библиотека с реализацией
add_library(nnue_core
    src/core.cpp
)

target_include_directories(nnue_core PUBLIC include)

# Основное приложение
add_executable(nnue_main
    main.cpp
)

target_link_libraries(nnue_main nnue_core)

# GoogleTest через FetchContent
include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# Тесты
add_executable(nnue_tests
    tests/main.cpp
    tests/test_calculate_input.cpp
)

target_link_libraries(nnue_tests
    nnue_core
    gtest_main
    pthread
)

include(GoogleTest)
gtest_discover_tests(nnue_tests)
