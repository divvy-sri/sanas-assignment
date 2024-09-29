cmake_minimum_required(VERSION 3.5)
project(LinearLayer LANGUAGES CXX)

# include(CTEST)
# enable_testing()

add_library(LinearLib linear.cpp linear.h)

add_executable(LinearMain main.cpp)

target_link_libraries(LinearMain PRIVATE LinearLib)

set_property(TARGET LinearMain PROPERTY CXX_STANDARD 17)
