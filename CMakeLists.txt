cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(PVAnalyzer)

find_package(OpenCV REQUIRED)

add_executable(PVAnalyzer forward.cpp main.cpp)

target_link_libraries(PVAnalyzer PRIVATE ${OpenCV_LIBS})