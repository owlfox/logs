#include <catch2/catch_test_macros.hpp>


#include "../src/ch2.p55.c"

TEST_CASE("cases of leet", "[1]") {
    
  Solution sol;
  SECTION("test case 1") {
    vector<int> in{2,7,11,15};
    vector<int> out = sol.twoSum(in, 9);
    vector<int> expected{0, 1};
    CHECK(expected == out);
  }
  
}