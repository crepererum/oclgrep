# generic CMake setup
cmake_minimum_required (VERSION 3.0)
include (ExternalProject)
set (CMAKE_EXPORT_COMPILE_COMMANDS ON)

# project metadata
project (oclgrep)

# compiler flags
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -g -gsplit-dwarf -pthread -Wall -Wextra -Wconversion -Wsign-conversion -Wpedantic -Wno-unknown-pragmas")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# detect clang
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-global-constructors -Wno-missing-prototypes -Wno-padded -Wno-shadow -fcolor-diagnostics -fdiagnostics-show-category=name")
endif ()

# faster linker
execute_process(COMMAND ${CMAKE_C_COMPILER} -fuse-ld=gold -Wl,--version ERROR_QUIET OUTPUT_VARIABLE LD_VERSION)
if ("${LD_VERSION}" MATCHES "GNU gold")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold -Wl,--disable-new-dtags")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold -Wl,--disable-new-dtags")
else ()
    message(WARNING "GNU gold linker isn't available, using the default system linker.")
endif ()

# bunch of external projects
externalproject_add (
    project_cmock
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/cmock"
    GIT_REPOSITORY "https://github.com/ThrowTheSwitch/CMock.git"
    GIT_TAG "8dc65e82bf4697a2fdc86f0d45ccad0749028d2d"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    TEST_COMMAND ""
)

externalproject_add (
    project_unity
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/unity"
    GIT_REPOSITORY "https://github.com/ThrowTheSwitch/Unity.git"
    GIT_TAG "v2.1.0"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    TEST_COMMAND ""
)

externalproject_add (
    project_clheaders
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/clheaders"
    GIT_REPOSITORY "https://github.com/KhronosGroup/OpenCL-Headers"
    GIT_TAG "a501078069186a4a950590382abb926b70114f89"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR> <INSTALL_DIR>/include/CL
    TEST_COMMAND ""
)

externalproject_add (
    project_clhpp
    DEPENDS project_cmock project_unity project_clheaders
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/clhpp"
    GIT_REPOSITORY "https://github.com/KhronosGroup/OpenCL-CLHPP.git"
    GIT_TAG "v2.0.9"
    PATCH_COMMAND sed -e "s/add_subdirectory(examples)//g" -e "s/add_subdirectory(tests)//g" -i "" CMakeLists.txt
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -DCMOCK_DIR=${CMAKE_CURRENT_BINARY_DIR}/cmock/src/cmock -DUNITY_DIR=${CMAKE_CURRENT_BINARY_DIR}/unity/src/unity -DOPENCL_INCLUDE_DIR=${CMAKE_CURRENT_BINARY_DIR}/clheaders/include
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target generate_cl2hpp
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <BINARY_DIR>/include/CL <INSTALL_DIR>/include/CL
    TEST_COMMAND ""
)

include_directories (
    SYSTEM
    "${CMAKE_CURRENT_BINARY_DIR}/clheaders/include"
    "${CMAKE_CURRENT_BINARY_DIR}/clhpp/include"
)

include_directories (
    "include"
)

# main executable
aux_source_directory ("src" SourceFiles)
add_executable (oclgrep ${SourceFiles})
add_dependencies (oclgrep project_clhpp)
target_link_libraries (
    oclgrep
    boost_locale
    boost_program_options
    OpenCL
)
