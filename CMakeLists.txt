cmake_minimum_required(VERSION 3.1)

project(rp)

find_package(OpenCV REQUIRED)

add_executable(rp main.cpp)

target_link_libraries(rp PRIVATE ${OpenCV_LIBS})