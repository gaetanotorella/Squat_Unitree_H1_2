cmake_minimum_required(VERSION 3.12)
project(h1_2_squat_controller)

# Il codice usa std::clamp e std::filesystem (C++17). gcc 11 (Ubuntu 22.04) compila in C++17 di
# default, gcc 9 (Ubuntu 20.04, il PC di bordo del robot) in C++14: senza questa riga la build sul
# robot fallisce con "'clamp' is not a member of 'std'". Trovato collaudando l'immagine Docker.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

# onnxruntime e' vendorizzato in deploy/thirdparty/ come tarball prebuilt, e il tarball e' per
# architettura: sul PC di sviluppo (x86_64) serve onnxruntime-linux-x64-1.22.0, sul PC di bordo
# del robot, se e' un Jetson (aarch64), serve onnxruntime-linux-aarch64-1.22.0 -- stessa versione,
# stesso layout include/ + lib/. Scelto qui in base a CMAKE_SYSTEM_PROCESSOR, cosi' lo stesso
# albero compila su entrambi senza ritoccare nulla. Tutto il resto (unitree_sdk2 + CycloneDDS,
# yaml-cpp, boost, eigen, spdlog/fmt) viene da `make install` del SDK e da apt.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(ONNXRUNTIME_DIR ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-aarch64-1.22.0)
else()
  set(ONNXRUNTIME_DIR ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0)
endif()
if(NOT EXISTS ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)
  message(FATAL_ERROR "onnxruntime non trovato per ${CMAKE_SYSTEM_PROCESSOR}: atteso ${ONNXRUNTIME_DIR}. "
                      "Scarica il tarball 1.22.0 per questa architettura da github.com/microsoft/onnxruntime/releases "
                      "e scompattalo in deploy/thirdparty/.")
endif()
message(STATUS "onnxruntime: ${ONNXRUNTIME_DIR}")

include_directories(
  /usr/include/eigen3
  /usr/local/include/ddscxx # dds
  /usr/local/include/iceoryx/v2.0.2 # dds
  include
  ${ONNXRUNTIME_DIR}/include
  ${PROJECT_SOURCE_DIR}/../../include/
)

link_libraries(
  unitree_sdk2 ddsc ddscxx rt pthread # dds
  libboost_program_options.a libyaml-cpp.a fmt
  ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
  stdc++fs   # std::filesystem su gcc 9 (Ubuntu 20.04); su gcc >= 11 e' uno stub innocuo
)

file(GLOB_RECURSE ADD_SRC_LIST
  ${PROJECT_SOURCE_DIR}/src/*.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
link_libraries(${PROJECT_NAME}_lib)

add_executable(h1_2_ctrl_squat main.cpp)