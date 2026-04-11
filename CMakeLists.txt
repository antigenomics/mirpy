cmake_minimum_required(VERSION 3.18)
project(mir_native LANGUAGES CXX)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# --- seqdist_c (mir.distances) — distances + CDR3 scoring ---
pybind11_add_module(seqdist_c MODULE mir/distances/seqdist.cpp)
target_compile_features(seqdist_c PRIVATE cxx_std_17)
if (MSVC)
  target_compile_options(seqdist_c PRIVATE /O2 /DNOMINMAX)
else()
  target_compile_options(seqdist_c PRIVATE -O3)
endif()
set_target_properties(seqdist_c PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mir/distances"
)
install(TARGETS seqdist_c LIBRARY DESTINATION mir/distances)

# --- mirseq (mir.basic) ---
pybind11_add_module(mirseq MODULE mir/basic/mirseq.cpp)
target_compile_features(mirseq PRIVATE cxx_std_17)
if (MSVC)
  target_compile_options(mirseq PRIVATE /O2 /DNOMINMAX)
else()
  target_compile_options(mirseq PRIVATE -O3)
endif()
set_target_properties(mirseq PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mir/basic"
)
install(TARGETS mirseq LIBRARY DESTINATION mir/basic)
