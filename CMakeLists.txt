cmake_minimum_required(VERSION 3.20)
project(metalrl CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

if(DEFINED ENV{METAL_CPP_DIR})
    set(METAL_CPP_DIR $ENV{METAL_CPP_DIR})
else()
    set(METAL_CPP_DIR "$ENV{HOME}/.local/include/metal-cpp")
endif()

# ── Every .metal file (framework, model, examples) links into one default.metallib ──
file(GLOB METAL_SHADERS
        "${CMAKE_SOURCE_DIR}/*.metal"
        "${CMAKE_SOURCE_DIR}/model/*.metal"
        "${CMAKE_SOURCE_DIR}/examples/*/*.metal"
)
set(AIR_FILES "")
foreach(SHADER ${METAL_SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    set(AIR_FILE "${CMAKE_BINARY_DIR}/${SHADER_NAME}.air")
    add_custom_command(
            OUTPUT ${AIR_FILE}
            COMMAND xcrun -sdk macosx metal -c -I ${CMAKE_SOURCE_DIR} ${SHADER} -o ${AIR_FILE}
            DEPENDS ${SHADER} ${CMAKE_SOURCE_DIR}/rollout.h ${CMAKE_SOURCE_DIR}/model/config.h
            COMMENT "Compiling Metal shader ${SHADER_NAME}.metal"
    )
    list(APPEND AIR_FILES ${AIR_FILE})
endforeach()
set(METALLIB "${CMAKE_BINARY_DIR}/default.metallib")
add_custom_command(
        OUTPUT ${METALLIB}
        COMMAND xcrun -sdk macosx metallib ${AIR_FILES} -o ${METALLIB}
        DEPENDS ${AIR_FILES}
        COMMENT "Linking default.metallib"
)
add_custom_target(metal_shaders DEPENDS ${METALLIB})

# ── One executable per example; each runs next to default.metallib ──
function(metalrl_example NAME SOURCE)
    add_executable(${NAME} ${SOURCE})
    target_include_directories(${NAME} PRIVATE ${CMAKE_SOURCE_DIR} ${METAL_CPP_DIR})
    target_link_libraries(${NAME} PRIVATE
            "-framework Metal"
            "-framework Foundation"
            "-framework QuartzCore"
    )
    add_dependencies(${NAME} metal_shaders)
endfunction()

metalrl_example(cartpole examples/CartPole/main.cpp)
metalrl_example(voxelnav examples/VoxelNav/main.cpp)
metalrl_example(cartpole_continuous examples/CartPoleContinuous/main.cpp)
metalrl_example(bench_cartpole bench/metalrl_bench.cpp)
metalrl_example(bench_voxelnav bench/metalrl_bench_voxelnav.cpp)
