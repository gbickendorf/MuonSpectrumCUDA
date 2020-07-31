cmake_minimum_required(VERSION 3.5.1)

# project name
project(MuonSpectrumCuda LANGUAGES CUDA CXX)

set(EXEC MuonSpectrumCuda)


if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(DCMAKE_CUDA_FLAGS -dc)

# I../includes
include_directories(includes)


# puts all .cpp files inside src to the SOURCES variable
file(GLOB SOURCES src/*.cu)

# compiles the files defined by SOURCES to generante the executable defined by EXEC
add_executable(${EXEC} ${SOURCES})
set_target_properties(${EXEC} PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
set_property(TARGET ${EXEC}  
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

