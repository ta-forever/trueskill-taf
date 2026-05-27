add_library(tafskill STATIC
    TafskillFactorGraph.h
    TafskillFactorGraph.cpp
    TafskillParams.h)

target_include_directories(tafskill PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(tafskill PUBLIC factorgraph)
