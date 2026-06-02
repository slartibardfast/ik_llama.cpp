# Find the nccl libraries
#
# The following variables are optionally searched for defaults
#  NCCL_ROOT: Base directory where all NCCL components are found
#  NCCL_INCLUDE_DIR: Directory where NCCL header is found
#  NCCL_LIB_DIR: Directory where NCCL library is found
#
# The following are set after configuration is done:
#  NCCL_FOUND
#  NCCL_INCLUDE_DIRS
#  NCCL_LIBRARIES
#
# The path hints include CUDA_TOOLKIT_ROOT_DIR seeing as some folks
# install NCCL in the same location as the CUDA toolkit.
# See https://github.com/caffe2/caffe2/issues/1601

set(NCCL_INCLUDE_DIR $ENV{NCCL_INCLUDE_DIR} CACHE PATH "Folder contains NVIDIA NCCL headers")
set(NCCL_LIB_DIR $ENV{NCCL_LIB_DIR} CACHE PATH "Folder contains NVIDIA NCCL libraries")
set(NCCL_VERSION $ENV{NCCL_VERSION} CACHE STRING "Version of NCCL to build with")

if ($ENV{NCCL_ROOT_DIR})
  message(WARNING "NCCL_ROOT_DIR is deprecated. Please set NCCL_ROOT instead.")
endif()
list(APPEND NCCL_ROOT $ENV{NCCL_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR})
# Compatible layer for CMake <3.12. NCCL_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
list(APPEND CMAKE_PREFIX_PATH ${NCCL_ROOT})

find_path(NCCL_INCLUDE_DIRS
  NAMES nccl.h
  HINTS ${NCCL_INCLUDE_DIR})

if (USE_STATIC_NCCL)
  MESSAGE(STATUS "USE_STATIC_NCCL is set. Linking with static NCCL library.")
  SET(NCCL_LIBNAME "nccl_static")
  if (NCCL_VERSION)  # Prefer the versioned library if a specific NCCL version is specified
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a.${NCCL_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
else()
  SET(NCCL_LIBNAME "nccl")
  if (NCCL_VERSION)  # Prefer the versioned library if a specific NCCL version is specified
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".so.${NCCL_VERSION}" ${CMAKE_FIND_LIBRARY_SUFFIXES})
  endif()
endif()

find_library(NCCL_LIBRARIES
  NAMES ${NCCL_LIBNAME}
  HINTS ${NCCL_LIB_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIRS NCCL_LIBRARIES)

if(NCCL_FOUND)  # obtaining NCCL version and some sanity checks
  set (NCCL_HEADER_FILE "${NCCL_INCLUDE_DIRS}/nccl.h")
  message (STATUS "Determining NCCL version from ${NCCL_HEADER_FILE}...")

  # Parse the version straight from the header. The previous approach used
  # check_cxx_symbol_exists(NCCL_VERSION_CODE) + a try_run, which false-negatives
  # whenever the bare symbol-check can't compile nccl.h (it transitively needs the
  # CUDA headers, absent from CMAKE_REQUIRED_INCLUDES here). That misreported a
  # modern NCCL (e.g. 2.30.4 -> NCCL_VERSION_CODE 23004) as "< 2.3.5-5". A header
  # string-parse is robust and needs no compile/link/run.
  file(STRINGS "${NCCL_HEADER_FILE}" _nccl_ver_defs
       REGEX "^[ \t]*#define[ \t]+NCCL_(MAJOR|MINOR|PATCH)[ \t]+[0-9]+")
  if (_nccl_ver_defs)
    string(REGEX MATCH "NCCL_MAJOR[ \t]+([0-9]+)" _ "${_nccl_ver_defs}")
    set(_nccl_major "${CMAKE_MATCH_1}")
    string(REGEX MATCH "NCCL_MINOR[ \t]+([0-9]+)" _ "${_nccl_ver_defs}")
    set(_nccl_minor "${CMAKE_MATCH_1}")
    string(REGEX MATCH "NCCL_PATCH[ \t]+([0-9]+)" _ "${_nccl_ver_defs}")
    set(_nccl_patch "${CMAKE_MATCH_1}")
    set(NCCL_VERSION_FROM_HEADER "${_nccl_major}.${_nccl_minor}.${_nccl_patch}")
    message(STATUS "NCCL version: ${NCCL_VERSION_FROM_HEADER}")
  else()
    message(STATUS "NCCL version: unknown (no NCCL_MAJOR define in ${NCCL_HEADER_FILE})")
  endif ()

  message(STATUS "Found NCCL (include: ${NCCL_INCLUDE_DIRS}, library: ${NCCL_LIBRARIES})")
  mark_as_advanced(NCCL_ROOT_DIR NCCL_INCLUDE_DIRS NCCL_LIBRARIES)
endif()

