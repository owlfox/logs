include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        devel #aka main branch
)
FetchContent_MakeAvailable(Catch2)

add_executable(unit_tests
               calc_test.cpp
               )
target_link_libraries(unit_tests PRIVATE
                      sut Catch2::Catch2WithMain)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(unit_tests)
