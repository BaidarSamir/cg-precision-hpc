cmake_minimum_required(VERSION 3.14)
project(cg_precision_hpc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O2 -Wall -Wextra -Wpedantic)
elseif(MSVC)
    add_compile_options(/O2 /W4)
endif()

add_executable(cg_solver src/main.cpp)
target_include_directories(cg_solver PRIVATE src)
