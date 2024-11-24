cmake_minimum_required(VERSION 3.31)
project(heat_equation)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Matplot++ REQUIRED)
find_package(MLX CONFIG REQUIRED)

add_executable(mlll main.cpp)
target_link_libraries(mlll PUBLIC Matplot++::matplot mlx)
target_include_directories(mlll PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
