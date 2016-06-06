project(NNTransitionSegmentor)

include_directories(
  basic/
  feature/
  model/
  state/
# please specify where you put mshadow.   
# for example, /users/mszhang/mshadow/
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


add_executable(LSTMSegmentor LSTMSegmentor.cpp)
add_executable(IntegratedLSTMSegmentor IntegratedLSTMSegmentor.cpp)
add_executable(LinearSegmentor LinearSegmentor.cpp)



target_link_libraries(LSTMSegmentor openblas)
target_link_libraries(IntegratedLSTMSegmentor openblas)
target_link_libraries(LinearSegmentor openblas)




#add_executable(TNNSegmentor TNNSegmentor.cpp)
#add_executable(GRNNSegmentor GRNNSegmentor.cpp)
#add_executable(RNNSegmentor RNNSegmentor.cpp)
#add_executable(LSTMNWSegmentor LSTMNWSegmentor.cpp)
#add_executable(LSTMNASegmentor LSTMNASegmentor.cpp)
#add_executable(LSTMNCSegmentor LSTMNCSegmentor.cpp)
#add_executable(LSTMNUCSegmentor LSTMNUCSegmentor.cpp)
#add_executable(LSTMNBCSegmentor LSTMNBCSegmentor.cpp)


#target_link_libraries(TNNSegmentor openblas)
#target_link_libraries(GRNNSegmentor openblas)
#target_link_libraries(RNNSegmentor openblas)
#target_link_libraries(LSTMNWSegmentor openblas)
#target_link_libraries(LSTMNASegmentor openblas)
#target_link_libraries(LSTMNCSegmentor openblas)
#target_link_libraries(LSTMNUCSegmentor openblas)
#target_link_libraries(LSTMNBCSegmentor openblas)


