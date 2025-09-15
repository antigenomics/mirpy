cmake_minimum_required(VERSION 3.18)
project(mir_cdrscore LANGUAGES CXX)

find_package(pybind11 CONFIG REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Модуль будет установлен как mir/distances/cdrscore*.so
pybind11_add_module(cdrscore MODULE mir/distances/cdrscore.cpp)
target_compile_features(cdrscore PRIVATE cxx_std_17)
# Релизные флаги
if (MSVC)
  target_compile_options(cdrscore PRIVATE /O2 /DNOMINMAX)
else()
  target_compile_options(cdrscore PRIVATE -O3 -march=native)
endif()

# Ставим в подпакет mir/distances
set_target_properties(cdrscore PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/mir/distances"
)
install(TARGETS cdrscore LIBRARY DESTINATION mir/distances)
