cmake_minimum_required(VERSION 3.10)

project(PiEstimation)

set(CMAKE_CXX_STANDARD 11)
if(CMAKE_BUILD_TYPE MATCHES "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
else(CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -pg")
endif()

add_executable(pi_estimate src/hw0.cxx)
