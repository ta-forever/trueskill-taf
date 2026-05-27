cmake_minimum_required(VERSION 3.18)
project(tafskill CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    add_compile_options(/W3 /EHsc /bigobj)
    add_compile_definitions(_USE_MATH_DEFINES NOMINMAX _CRT_SECURE_NO_WARNINGS)
    # Eigen 3.2 at D:/wrk/eigen uses std::binder1st / std::pointer_to_unary_function
    # which were removed in C++17. Building as C++14 keeps them available.
    add_compile_definitions(_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Eigen
if(NOT DEFINED EIGEN3_INCLUDE_DIR)
    set(EIGEN3_INCLUDE_DIR "D:/wrk/eigen/eigen-eigen-f562a193118d" CACHE PATH "Eigen3 include dir")
endif()
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(${CMAKE_SOURCE_DIR}/src)

add_subdirectory(src/factorgraph)
add_subdirectory(src/tafskill)
add_subdirectory(src/pytafskill)
