cmake_minimum_required(VERSION 3.22...3.30)
project(g1_pilot)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(CMakeDependentOption)

add_compile_options(-Wall -Wextra -Wpedantic)

# ---- Integration & Path Logic ---- #

if(DEFINED ENV{AMENT_PREFIX_PATH} OR DEFINED ENV{COLCON_PREFIX_PATH} OR DEFINED ENV{ROS_DISTRO})
    set(ROS2_DEFAULT ON)
    message(STATUS "Building with ROS2")
else()
    set(ROS2_DEFAULT OFF)
    message(WARNING "NOT building with ROS2")
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install" CACHE PATH "Local install prefix" FORCE)
    endif()
endif()

# list(APPEND CMAKE_PREFIX_PATH /opt/openrobots)
# set(CMAKE_INSTALL_RPATH "/opt/openrobots/lib")
# set(CMAKE_BUILD_RPATH "/opt/openrobots/lib")

# ---- Options & Dependencies ---- #

find_package(casadi REQUIRED)
find_package(pinocchio REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(ompl REQUIRED)

option(BUILD_TESTING "Build tests" ON)
cmake_dependent_option(INSTALL_TESTING "Install test sources" OFF "BUILD_TESTING" ON)
cmake_dependent_option(RUN_TESTING "Run test scripts" BUILD_TESTING "NOT ROS2_DEFAULT" OFF)



# ---- Library ---- #

add_library(g1_pilot
    src/core/humanoid.cpp
    src/core/io.cpp
    src/core/arm_ik/robot_arm_ik.cpp
    src/core/arm_control/arm_control.cpp
    src/core/arm_control/ik_traj_tracker.cpp
    src/core/arm_control/impedance_control.cpp
    src/core/arm_mp/motion_planner.cpp
    src/core/arm_mp/joint_space_planner.cpp
    src/core/arm_mp/visual_servo_planner.cpp
    src/core/body_ekf/body_ekf.cpp
)

target_compile_definitions(g1_pilot PUBLIC COAL_DISABLE_HPP_FCL_WARNINGS)

target_include_directories(g1_pilot 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${OMPL_INCLUDE_DIRS}
)

target_link_libraries(g1_pilot 
    PUBLIC
        Eigen3::Eigen
    PRIVATE 
        pinocchio::pinocchio 
        casadi::casadi 
        yaml-cpp::yaml-cpp 
        ${OMPL_LIBRARIES}
)

# ---- Testing ---- #

if(BUILD_TESTING)
    find_package(GTest REQUIRED)

    add_executable(test_brake test/arm_mp/test_brake.cpp)
    target_include_directories(test_brake PRIVATE src include)
    target_link_libraries(test_brake PRIVATE g1_pilot pinocchio::pinocchio GTest::gtest_main)

    add_executable(benchmark_brake test/arm_mp/benchmark_brake.cpp)
    target_include_directories(benchmark_brake PRIVATE src include)
    target_link_libraries(benchmark_brake PRIVATE g1_pilot pinocchio::pinocchio)
    
    add_executable(test_ik test/arm_ik/test_ik.cpp)
    target_link_libraries(test_ik PRIVATE g1_pilot GTest::gtest_main)

    add_executable(test_control test/arm_control/test_control.cpp)
    target_link_libraries(test_control PRIVATE g1_pilot GTest::gtest_main)

    add_executable(test_mp_ik test/arm_mp/test_traj_gen.cpp)
    target_link_libraries(test_mp_ik PRIVATE g1_pilot GTest::gtest_main)
    # The default RobotConfig paths are relative and only resolve from one
    # specific working directory; give the tests absolute ones instead.
    target_compile_definitions(test_mp_ik PRIVATE
        G1_ASSET_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../g1_description/assets/g1/"
        G1_CONFIG_FILE="${CMAKE_CURRENT_SOURCE_DIR}/config/g1.yaml")

    add_executable(test_body_ekf test/body_ekf/test_body_ekf.cpp)
    target_include_directories(test_body_ekf PRIVATE src)
    target_link_libraries(test_body_ekf PRIVATE g1_pilot GTest::gtest_main)
    target_compile_definitions(test_body_ekf PRIVATE
        G1_ASSET_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../g1_description/assets/g1/"
        G1_CONFIG_FILE="${CMAKE_CURRENT_SOURCE_DIR}/config/g1.yaml")

    add_executable(body_ekf_replay test/body_ekf/replay_body_ekf.cpp)
    target_include_directories(body_ekf_replay PRIVATE src)
    target_link_libraries(body_ekf_replay PRIVATE g1_pilot)

    # add_executable(math_unittest test/arm_mp/math_unittest.cpp)
    # add_executable(line_unittest test/arm_mp/line_unittest.cpp)
    # target_link_libraries(math_unittest PRIVATE arm_mp GTest::GTest GTest::Main pthread)
    # target_link_libraries(line_unittest PRIVATE arm_mp GTest::GTest GTest::Main pthread)

    if (RUN_TESTING)
        include(GoogleTest)
        gtest_discover_tests(test_brake)
        gtest_discover_tests(test_ik WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)
        gtest_discover_tests(test_control WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)
        # gtest_discover_tests(math_unittest WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)
        # gtest_discover_tests(line_unittest WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)
        gtest_discover_tests(test_mp_ik WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test)
        gtest_discover_tests(test_body_ekf)
    endif()


endif()

# --- Installation --- #

include(GNUInstallDirs)

install(TARGETS 
    g1_pilot 
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(INSTALL_TESTING)
    install(FILES test/arm_ik/test_ik.cpp DESTINATION share/${PROJECT_NAME}/test)
    install(FILES test/arm_control/test_control.cpp DESTINATION share/${PROJECT_NAME}/test)
endif()

# ---- ROS 2 Integration ---- #

if(ROS2_DEFAULT)
    find_package(ament_cmake REQUIRED)

    set(ROS2_DEPS
        ament_index_cpp
        rclcpp
        rclcpp_action
        sensor_msgs
        nav_msgs
        geometry_msgs
        trajectory_msgs
        tf2_ros
        tf2_eigen
        tf2_geometry_msgs
        std_srvs
        humanoid_manipulation_interfaces
    )

    foreach(dep ${ROS2_DEPS})
        find_package(${dep} REQUIRED)
    endforeach()

    add_executable(ik_joint_state_publisher src/joint_state_publisher.cpp)
    add_executable(impede_arms src/impede_arms.cpp)
    add_executable(joint_traj_publisher src/joint_traj_publisher.cpp)
    add_executable(visual_servo src/visual_servo.cpp)
    add_executable(grav_ff src/grav_ff.cpp)
    add_executable(body_ekf src/body_ekf.cpp)
    target_include_directories(body_ekf PRIVATE src)

    # Service Servers
    add_executable(plan_trajecory src/plan_trajecory.cpp)
    add_executable(ik_server src/ik_server.cpp)

    # Action Servers
    add_executable(joint_controller src/joint_controller.cpp)
    add_executable(move_to_pose
        src/move_to_pose.cpp
        src/move_to_pose/server.cpp
        src/move_to_pose/planning.cpp
        src/move_to_pose/control.cpp
    )

    target_link_libraries(ik_joint_state_publisher g1_pilot)
    ament_target_dependencies(ik_joint_state_publisher ${ROS2_DEPS})

    target_link_libraries(impede_arms g1_pilot)
    ament_target_dependencies(impede_arms ${ROS2_DEPS})

    target_link_libraries(joint_traj_publisher g1_pilot)
    ament_target_dependencies(joint_traj_publisher ${ROS2_DEPS})

    target_link_libraries(visual_servo g1_pilot)
    ament_target_dependencies(visual_servo ${ROS2_DEPS})

    target_link_libraries(grav_ff g1_pilot)
    ament_target_dependencies(grav_ff ${ROS2_DEPS})

    target_link_libraries(body_ekf g1_pilot)
    ament_target_dependencies(body_ekf ${ROS2_DEPS})

    target_link_libraries(joint_controller g1_pilot)
    ament_target_dependencies(joint_controller ${ROS2_DEPS})

    target_link_libraries(plan_trajecory g1_pilot)
    ament_target_dependencies(plan_trajecory ${ROS2_DEPS})

    target_link_libraries(ik_server g1_pilot)
    ament_target_dependencies(ik_server ${ROS2_DEPS})

    target_link_libraries(move_to_pose g1_pilot)
    ament_target_dependencies(move_to_pose ${ROS2_DEPS})

    install(TARGETS
        ik_joint_state_publisher
        impede_arms
        joint_traj_publisher
        visual_servo
        grav_ff
        body_ekf
        plan_trajecory
        ik_server
        joint_controller
        move_to_pose
        DESTINATION lib/${PROJECT_NAME}
    )

    install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
    install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

    if(BUILD_TESTING)
        find_package(ament_lint_auto REQUIRED)
        ament_lint_auto_find_test_dependencies()
    endif()

    ament_package()
endif()
