cmake_minimum_required(VERSION 3.16)
project(nnue_project LANGUAGES CXX)

# Устанавливаем стандарт языка (например, C++20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # для clangd/IDE

# Опционально: Treat warnings as errors
# add_compile_options(-Wall -Wextra -Werror)

# Определяем главный include path для публичных заголовков
include_directories(${PROJECT_SOURCE_DIR}/include)

# === Добавляем основную библиотеку NNUE ===
add_library(nnue
    src/nnue.cpp
    src/layers.cpp
    src/utils.cpp
)

target_include_directories(nnue
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(nnue PUBLIC cxx_std_20)

# === Добавляем модуль обучения ===
add_library(nnue_train
    train/src/train.cpp
    train/src/dataset.cpp
)

target_include_directories(nnue_train
    PUBLIC
        ${PROJECT_SOURCE_DIR}/train/include
        ${PROJECT_SOURCE_DIR}/include  # чтобы видеть NNUE API
)

target_link_libraries(nnue_train
    PRIVATE nnue
)

# === Пример использования ===
add_executable(run_nnue
    examples/run_nnue.cpp
)

target_link_libraries(run_nnue
    PRIVATE nnue
)

# === Тесты (GoogleTest или Catch2) ===
enable_testing()

# Здесь предположим, что ты используешь Catch2 (header-only)
add_executable(tests
    tests/test_nnue.cpp
    tests/test_layers.cpp
    tests/test_train.cpp
)

target_link_libraries(tests
    PRIVATE nnue nnue_train
)

add_test(NAME all_tests COMMAND tests)

# === Установка (если нужно делать install / reuse как библиотеку) ===
install(TARGETS nnue nnue_train
    EXPORT nnueTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

# === Экспортируем для внешнего использования ===
export(EXPORT nnueTargets
    FILE ${CMAKE_BINARY_DIR}/nnueTargets.cmake
    NAMESPACE nnue::
)
