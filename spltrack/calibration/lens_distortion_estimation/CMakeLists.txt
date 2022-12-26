add_library(ami_pol)
target_sources(ami_pol 
    PRIVATE 
        ami_pol.c
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/ami_pol.h
)

target_include_directories(ami_pol PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(lens_distortion_estimation)

target_sources(lens_distortion_estimation PRIVATE 
    lens_distortion_estimation.cpp
    lens_distortion.cpp
    lens_distortion.h
    point2d.h
    image/image.h
    image/ami_bmp/ami_bmp.cpp
    image/ami_bmp/ami_bmp.h
)

target_include_directories(lens_distortion_estimation PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/image

)

target_compile_options(lens_distortion_estimation PRIVATE
    -Wall 
    -Wextra
    -O3 
    -funroll-loops 
    -fomit-frame-pointer 
)

target_link_libraries(lens_distortion_estimation
    PUBLIC
        ami_pol
        OpenMP::OpenMP_CXX
)

set_target_properties(lens_distortion_estimation
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)
