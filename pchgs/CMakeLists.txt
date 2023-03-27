cmake_minimum_required(VERSION 3.14)

include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)

project(PCHGS VERSION 0.1 LANGUAGES CXX)

include(cmake/CPM.cmake)
include(cmake/tools.cmake)

add_subdirectory(lib)

option(SUBMISSION_MODE "Build in submission mode." TRUE)

file(GLOB_RECURSE headers CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
file(GLOB_RECURSE sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(${PROJECT_NAME} ${headers} ${sources})
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)

target_link_libraries(${PROJECT_NAME} PRIVATE fmt::fmt)
target_link_libraries(${PROJECT_NAME} PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(${PROJECT_NAME} PRIVATE sul::dynamic_bitset)

target_include_directories(
        ${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

target_compile_options(${PROJECT_NAME} PUBLIC "-Wall" "-Wextra" "-Wpedantic")
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:DEBUG>:-O0;-g3;-ggdb>")
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:RELEASE>:-O3;-s;-march=native;-mtune=native;-flto>")
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:RELWITHDEBINFO>:-O3;-march=native;-g3;-ggdb;-fno-inline>")
target_link_options(${PROJECT_NAME} PUBLIC "-flto")

set_target_properties(${PROJECT_NAME} PROPERTIES INTERPROCEDUAL_OPTIMIZATION $<$<CONFIG:RELEASE>:TRUE>)

if (SUBMISSION_MODE)
    message("Building in submission mode")
    target_compile_definitions(${PROJECT_NAME} PUBLIC "SUBMISSION_MODE=TRUE")
endif ()
