file(GLOB SOURCE_FILES *.h *.cpp)
add_library(factorgraph STATIC ${SOURCE_FILES})
target_include_directories(factorgraph PUBLIC ${CMAKE_SOURCE_DIR}/src)
