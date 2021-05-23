cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(PVAnalyzer)

find_package(OpenCV REQUIRED)
find_package (PythonLibs REQUIRED)

add_executable(PVAnalyzer forward.cpp main.cpp)

include_directories(${PYTHON_INCLUDE_DIRS})

target_link_libraries(PVAnalyzer PUBLIC ${OpenCV_LIBS})
target_link_libraries(PVAnalyzer PUBLIC ${PYTHON_LIBRARIES})
