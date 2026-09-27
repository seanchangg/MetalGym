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

add_executable(metalRL main.cpp)

target_include_directories(metalRL PRIVATE ${METAL_CPP_DIR})

target_link_libraries(metalRL PRIVATE
        "-framework Metal"
        "-framework MetalPerformanceShaders"
        "-framework Foundation"
        "-framework QuartzCore"
)

# ── Compile the .metal shaders into default.metallib at build time ──
file(GLOB METAL_SHADERS
        "${CMAKE_SOURCE_DIR}/*.metal"
        "${CMAKE_SOURCE_DIR}/model/*.metal"
)
if(METAL_SHADERS)
    set(AIR_FILES "")
    foreach(SHADER ${METAL_SHADERS})
        get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
        set(AIR_FILE "${CMAKE_BINARY_DIR}/${SHADER_NAME}.air")
        add_custom_command(
                OUTPUT ${AIR_FILE}
                COMMAND xcrun -sdk macosx metal -c ${SHADER} -o ${AIR_FILE}
                DEPENDS ${SHADER}
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
    add_dependencies(metalRL metal_shaders)
endif()
