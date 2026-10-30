cmake_minimum_required(VERSION 3.8)
project(nav_dev)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# 実行ファイルの作成と依存パッケージの紐付け
add_executable(teleop_node src/teleop_node.cpp)
ament_target_dependencies(teleop_node rclcpp geometry_msgs key_event_msgs key_event_nodes)

# 実行ファイルのインストール先を指定（これが無いとros2 runで見つかりません）
install(TARGETS
  teleop_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(FILES world.sdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()