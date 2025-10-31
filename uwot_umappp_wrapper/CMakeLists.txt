cmake_minimum_required(VERSION 3.16)
project(uwot_umappp_wrapper)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include FetchContent module for minimal dependencies
include(FetchContent)

# Suppress all Eigen-related warnings and C++11 issues before fetching
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Don't build Eigen tests" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "Don't build Eigen documentation" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Don't build any tests" FORCE)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON CACHE BOOL "Suppress developer warnings" FORCE)

# Fetch umappp dependencies
# Eigen is required by irlba and umappp spectral initialization
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)

FetchContent_Declare(
  aarand
  GIT_REPOSITORY https://github.com/LTLA/aarand
  GIT_TAG master
)

FetchContent_Declare(
  sanisizer
  GIT_REPOSITORY https://github.com/LTLA/sanisizer
  GIT_TAG master
)

FetchContent_Declare(
  subpar
  GIT_REPOSITORY https://github.com/LTLA/subpar
  GIT_TAG master
)

FetchContent_Declare(
  irlba
  GIT_REPOSITORY https://github.com/LTLA/CppIrlba
  GIT_TAG master
)

FetchContent_Declare(
  knncolle
  GIT_REPOSITORY https://github.com/knncolle/knncolle
  GIT_TAG master
)

# Make dependencies available
FetchContent_MakeAvailable(eigen)
FetchContent_MakeAvailable(aarand)
FetchContent_MakeAvailable(sanisizer)
FetchContent_MakeAvailable(subpar)
FetchContent_MakeAvailable(irlba)
FetchContent_MakeAvailable(knncolle)

# Find and enable OpenMP for parallel optimization
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
else()
    message(WARNING "OpenMP not found - umappp will run in single-threaded mode")
endif()

# Additional Eigen configuration (moved to top before FetchContent)

# Enable parallel optimization for better performance
# (umappp will use OpenMP when available)

# Add the wrapper library with existing source files
add_library(uwot_umappp SHARED
    uwot_simple_wrapper.cpp
    uwot_fit.cpp
    uwot_transform.cpp
    uwot_model.cpp
    uwot_persistence.cpp
    uwot_hnsw_utils.cpp
    uwot_progress_utils.cpp
    uwot_distance.cpp
    uwot_crc32.cpp
    lz4.c
)

# Include directories
target_include_directories(uwot_umappp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${aarand_SOURCE_DIR}/include
    ${sanisizer_SOURCE_DIR}/include
    ${subpar_SOURCE_DIR}/include
    ${irlba_SOURCE_DIR}/include
    ${knncolle_SOURCE_DIR}/include
    ${eigen_SOURCE_DIR}
)

# Link OpenMP for parallel optimization
if(OpenMP_CXX_FOUND)
    target_link_libraries(uwot_umappp PUBLIC OpenMP::OpenMP_CXX)
endif()

# Windows-specific DLL export settings
if(WIN32)
    target_compile_definitions(uwot_umappp PRIVATE UWOT_EXPORTS)
    set_target_properties(uwot_umappp PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )
endif()

# Set output name
set_target_properties(uwot_umappp PROPERTIES
    OUTPUT_NAME "uwot"
)

# Find and enable OpenMP if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(uwot_umappp PRIVATE OpenMP::OpenMP_CXX)
endif()

# Compiler-specific optimizations and warning suppression
if(MSVC)
    # High warning level for our code, but suppress unrelated warnings
    target_compile_options(uwot_umappp PRIVATE /W4 /O2 /fp:fast)
    # Suppress external library warnings for clean build
    target_compile_options(uwot_umappp PRIVATE /wd4244)  # conversion potential loss (already fixed in hnswalg.h)
    target_compile_options(uwot_umappp PRIVATE /wd4267)  # size_t to int conversion
    target_compile_options(uwot_umappp PRIVATE /wd4018)  # signed/unsigned mismatch
    target_compile_options(uwot_umappp PRIVATE /wd4996)  # deprecated functions
    target_compile_options(uwot_umappp PRIVATE /wd4100)  # unreferenced parameters (external libs)
    target_compile_options(uwot_umappp PRIVATE /wd4127)  # conditional expression is constant
    target_compile_options(uwot_umappp PRIVATE /wd4457)  # variable hiding
    target_compile_options(uwot_umappp PRIVATE /wd4245)  # signed/unsigned mismatch
    target_compile_options(uwot_umappp PRIVATE /wd4505)  # unreferenced function
    target_compile_options(uwot_umappp PRIVATE /wd4714)  # optimization related
else()
    # Maximum performance optimizations for GCC/Clang
    target_compile_options(uwot_umappp PRIVATE -Wall -Wextra -O3 -march=native -ffast-math)
    # Suppress common external library warnings
    target_compile_options(uwot_umappp PRIVATE -Wno-unused-parameter)
    target_compile_options(uwot_umappp PRIVATE -Wno-deprecated-declarations)
    target_compile_options(uwot_umappp PRIVATE -Wno-sign-compare)
    target_compile_options(uwot_umappp PRIVATE -Wno-unused-function)
endif()