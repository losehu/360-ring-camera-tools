cmake_minimum_required(VERSION 3.10)
project(MyProject)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 让 CMake 容易找到 Homebrew 前缀
list(PREPEND CMAKE_PREFIX_PATH "/opt/homebrew")
set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)

# 方法1：尝试使用 CONFIG 模式查找 Abseil
find_package(absl QUIET CONFIG)

# 方法2：如果 CONFIG 模式失败，尝试 MODULE 模式
if(NOT absl_FOUND)
    find_package(absl QUIET MODULE)
endif()

# 方法3：如果仍然找不到，尝试手动设置
if(NOT absl_FOUND)
    message(STATUS "Abseil not found via find_package, trying manual setup...")
    
    # 查找 Abseil 头文件和库文件
    find_path(ABSL_INCLUDE_DIR absl/base/config.h
        PATHS /opt/homebrew/include
    )
    
    find_library(ABSL_BASE_LIB absl_base
        PATHS /opt/homebrew/lib
    )
    
    find_library(ABSL_STRINGS_LIB absl_strings
        PATHS /opt/homebrew/lib
    )
    
    find_library(ABSL_LOG_LIB absl_log_internal_check_op
        PATHS /opt/homebrew/lib
    )
    
    if(ABSL_INCLUDE_DIR AND ABSL_BASE_LIB AND ABSL_STRINGS_LIB AND ABSL_LOG_LIB)
        message(STATUS "Found Abseil manually: ${ABSL_INCLUDE_DIR}")
        add_library(absl::base INTERFACE IMPORTED)
        set_target_properties(absl::base PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${ABSL_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "${ABSL_BASE_LIB}"
        )
        
        add_library(absl::strings INTERFACE IMPORTED)
        set_target_properties(absl::strings PROPERTIES
            INTERFACE_LINK_LIBRARIES "${ABSL_STRINGS_LIB}"
        )
        
        add_library(absl::log_internal_check_op INTERFACE IMPORTED)
        set_target_properties(absl::log_internal_check_op PROPERTIES
            INTERFACE_LINK_LIBRARIES "${ABSL_LOG_LIB}"
        )
        
        set(absl_FOUND TRUE)
    endif()
endif()

# 其他包的查找...
find_package(JsonCpp QUIET CONFIG)
if (NOT JsonCpp_FOUND)
  find_package(jsoncpp REQUIRED CONFIG)
endif()

if (TARGET jsoncpp_lib AND NOT TARGET JsonCpp::JsonCpp)
  add_library(JsonCpp::JsonCpp INTERFACE IMPORTED)
  set_target_properties(JsonCpp::JsonCpp PROPERTIES
    INTERFACE_LINK_LIBRARIES jsoncpp_lib)
endif()

# 重新查找 Boost
find_package(Boost 1.89.0 REQUIRED COMPONENTS  thread )
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(PCL REQUIRED)
find_package(Threads REQUIRED)

# 更新 COMMON_LIBS
set(COMMON_LIBS
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    ${Eigen3_LIBS}
    ${Boost_LIBRARIES}
    ${Threads_LIBRARIES}

)

# 只在找到 Abseil 时添加链接
if(absl_FOUND)
    if(TARGET absl::base)
        list(APPEND COMMON_LIBS absl::base absl::strings absl::log_internal_check_op)
    else()
        # 如果目标是手动创建的
        list(APPEND COMMON_LIBS ${ABSL_BASE_LIB} ${ABSL_STRINGS_LIB} ${ABSL_LOG_LIB})
    endif()
else()
    message(WARNING "Abseil not found, linking may fail")
endif()

function(add_my_executable target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE ${COMMON_LIBS})
    
    # 添加包含目录（如果手动设置了 Abseil）
    if(ABSL_INCLUDE_DIR)
        target_include_directories(${target_name} PRIVATE ${ABSL_INCLUDE_DIR})
    endif()
endfunction()

add_my_executable(main_pano main_pano.cpp)
add_my_executable(main      main.cpp)
add_my_executable(pcd_show  pcd_show.cpp)
add_my_executable(unfold  ../unfold.cpp)
add_my_executable(projectpano  projectCloud_pano.cpp)
add_my_executable(projectpano2  projectCloud_pano2.cpp)
add_my_executable(yuyan  main_yuyan.cpp)
add_my_executable(yuyan_show  projectCloud_yuyan.cpp)
add_my_executable(pano_yaw  main_pano_yaw.cpp)