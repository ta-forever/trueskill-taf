find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)

Python_add_library(pytafskill MODULE pytafskill.cpp)

target_include_directories(pytafskill PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${Python_INCLUDE_DIRS}
    ${Python_NumPy_INCLUDE_DIRS})

target_link_libraries(pytafskill PRIVATE tafskill factorgraph)

# Output as pytafskill.pyd on Windows (Python_add_library handles MODULE naming)
if (WIN32)
    set_target_properties(pytafskill PROPERTIES SUFFIX ".pyd")
endif()
