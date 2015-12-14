project(OpenSentiment)

include_directories(
  basic/
  feature/
  model/
  state/
  /opt/mshadow/
  ../LibN3L/
)

add_definitions(-DUSE_CUDA=0)

IF(CMAKE_BUILD_TYPE MATCHES Debug)
    SET( CMAKE_CXX_FLAGS  "-w -msse3 -funroll-loops -O0" )
ELSE()
    SET( CMAKE_CXX_FLAGS  "-w -msse3 -funroll-loops -O3" )
ENDIF()
####for   openblas
add_definitions(-DMSHADOW_USE_CUDA=0)
add_definitions(-DMSHADOW_USE_CBLAS=1)
add_definitions(-DMSHADOW_USE_MKL=0)

SET( CMAKE_SHARED_LINKER_FLAGS  "-lm -lopenblas")
####endfor openblas

####for   cuda
#add_definitions(-DMSHADOW_USE_CUDA=1)
#add_definitions(-DMSHADOW_USE_CBLAS=1)
#add_definitions(-DMSHADOW_USE_MKL=0)

#SET( CMAKE_SHARED_LINKER_FLAGS  "-lm -lcudart -lcublas -lcurand" )
#include_directories(
#  $(USE_CUDA_PATH)/include
#)
#LINK_DIRECTORIES($(USE_CUDA_PATH)/lib64)
####endfor cuda

#add_subdirectory(basic)

#aux_source_directory(. DIR_SRCS)


add_executable(StackLSTMSegmentor StackLSTMSegmentor.cpp)
add_executable(LinearSegmentor LinearSegmentor.cpp)
add_executable(APSegmentor APSegmentor.cpp)

target_link_libraries(StackLSTMSegmentor openblas)
target_link_libraries(LinearSegmentor openblas)
target_link_libraries(APSegmentor openblas)


