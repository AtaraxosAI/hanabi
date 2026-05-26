CMAKE_MINIMUM_REQUIRED(VERSION 3.15)
project(hanalearn)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-register -fPIC -Wfatal-errors")

execute_process(
  COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/get_pybind_flags.py
  OUTPUT_VARIABLE PYBIND_FLAGS
)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PYBIND_FLAGS}")
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hanabi-learning-environment)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11)

# find_package(PythonInterp 3.7 REQUIRED)
# find_package(PythonLibs 3.7 REQUIRED)

# find pytorch
execute_process(
  COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
  OUTPUT_VARIABLE TorchPath
)
list(APPEND CMAKE_PREFIX_PATH ${TorchPath})
find_package(Torch REQUIRED)
set(TORCH_PYTHON_LIBRARIES "${TorchPath}/lib/libtorch_python${CMAKE_SHARED_LIBRARY_SUFFIX}")

message("---------------------")
message(${TorchPath})
message(${PYBIND_FLAGS})
message(${CMAKE_CXX_FLAGS})
message("---------------------")


# lib for other infra programs
add_library(infra_lib
  cpp/infra/transition.cc
  cpp/infra/batcher.cc
  cpp/infra/batch_runner.cc
  cpp/infra/context.cc
  cpp/infra/episode_buffer.cc
)
target_include_directories(infra_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(infra_lib PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(infra_lib PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(infra_lib PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARIES})

pybind11_add_module(infra cpp/infra/pybind.cc)
target_include_directories(infra PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(infra PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(infra PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(infra PUBLIC infra_lib ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARIES})


# final hanalearn lib
pybind11_add_module(
  hanalearn
  # rl
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/utils.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/rl/rl_actor.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/rl/thread_loop.cc
  # search
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/search/game_sim.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/search/player.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/search/search.cc
  # pybind
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/rl/pybind.cc
)
target_link_libraries(hanalearn PUBLIC hanabi)
target_link_libraries(hanalearn PUBLIC infra_lib)
target_include_directories(hanalearn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
