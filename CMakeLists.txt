cmake_minimum_required(VERSION 3.1.0)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


# set root of location to find PETSc's pkg-config and slepc's pkg-config
set(PETSC $ENV{PETSC_DIR}/$ENV{PETSC_ARCH})
set(ENV{PKG_CONFIG_PATH} ${PETSC}/lib/pkgconfig)

set(SLPEC $ENV{SLEPC_DIR}/$ENV{PETSC_ARCH})
set(ENV{PKG_CONFIG_PATH} $ENV{PKG_CONFIG_PATH}:${SLPEC}/lib/pkgconfig)

message(STATUS "source path: ${PETSC}")
message(STATUS "source path: ${SLPEC}")

project(doctor)

set(targets TestLobpcg)

foreach(x IN LISTS targets)
    add_executable(${x} ${x}.cpp)
endforeach()

find_package(PkgConfig REQUIRED)
pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc)
pkg_search_module(SLEPC REQUIRED IMPORTED_TARGET slepc)

foreach(x IN LISTS targets)
    target_link_libraries(${x} PkgConfig::PETSC)
    target_link_libraries(${x} PkgConfig::SLEPC)
endforeach()