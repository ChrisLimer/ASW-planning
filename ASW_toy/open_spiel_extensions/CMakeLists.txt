cmake_minimum_required(VERSION 3.16)
project(asw_extension_plugin CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Your actual layout:
#   ASW_toy/open_spiel/build/libopen_spiel.so
#   ASW_toy/open_spiel/open_spiel/policy.h
set(OPEN_SPIEL_ROOT   "${CMAKE_SOURCE_DIR}/../open_spiel")
set(OPEN_SPIEL_LIBDIR "${OPEN_SPIEL_ROOT}/build")

add_subdirectory(
  "${OPEN_SPIEL_ROOT}/open_spiel/abseil-cpp"
  "${CMAKE_BINARY_DIR}/abseil"
  EXCLUDE_FROM_ALL
)

if(NOT EXISTS "${OPEN_SPIEL_LIBDIR}/libopen_spiel.so")
  message(FATAL_ERROR "Missing ${OPEN_SPIEL_LIBDIR}/libopen_spiel.so")
endif()

add_library(open_spiel SHARED IMPORTED GLOBAL)
set_target_properties(open_spiel PROPERTIES
  IMPORTED_LOCATION "${OPEN_SPIEL_LIBDIR}/libopen_spiel.so"
)

add_library(asw_extensions SHARED
  open_spiel/games/asw_small/asw_small.cc
  open_spiel/algorithms/json_policy.cc
  open_spiel/algorithms/get_all_info_and_terminal_states.cc
)

target_include_directories(asw_extensions PRIVATE
  "${OPEN_SPIEL_ROOT}"                    # so <open_spiel/...> resolves
  "${OPEN_SPIEL_ROOT}/open_spiel/json/include"  # nlohmann json (if needed)
  "${OPEN_SPIEL_ROOT}/open_spiel/abseil-cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}"           # so your extension headers resolve
)

target_link_libraries(asw_extensions PRIVATE open_spiel)


set_target_properties(asw_extensions PROPERTIES
  BUILD_RPATH "${OPEN_SPIEL_LIBDIR}"
)

## Example
add_executable(test_load_asw_small test_load_asw_small.cc)

target_include_directories(test_load_asw_small PRIVATE
  "${OPEN_SPIEL_ROOT}"
  "${OPEN_SPIEL_ROOT}/open_spiel/json/include"
  "${OPEN_SPIEL_ROOT}/open_spiel/abseil-cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(test_load_asw_small PRIVATE
  open_spiel
  asw_extensions
  dl
  absl::flags
  absl::flags_parse
)

set_target_properties(test_load_asw_small PROPERTIES
  BUILD_RPATH "${OPEN_SPIEL_LIBDIR};${CMAKE_BINARY_DIR}"
)


## Pub_amg_exploitability
add_executable(pub_amg_exploitability open_spiel/examples/pub_amg_exploitability.cc)

target_include_directories(pub_amg_exploitability PRIVATE
  "${OPEN_SPIEL_ROOT}"
  "${OPEN_SPIEL_ROOT}/open_spiel/json/include"
  "${OPEN_SPIEL_ROOT}/open_spiel/abseil-cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(pub_amg_exploitability PRIVATE
  open_spiel
  asw_extensions
  dl
  absl::flags
  absl::flags_parse
)

set_target_properties(pub_amg_exploitability PROPERTIES
  BUILD_RPATH "${OPEN_SPIEL_LIBDIR};${CMAKE_BINARY_DIR}"
)