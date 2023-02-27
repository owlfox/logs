#include <catch2/catch_test_macros.hpp>

extern "C" {
  #include "../src/calc.h"
}

//or alternatively
//#include "../src/calc.c"

TEST_CASE("SumAddsTwoInts", "[calc]") {

  CHECK(4 == sum(2, 2));
}

TEST_CASE("MultiplyMultipliesTwoInts", "[calc]") {

  CHECK(12 == multiply(3, 4));
}
