#include "catch.hpp"
#include "test_fixture.hpp"

TEST_CASE_METHOD(array_fixture, "array_is_not_empty" ) {
  REQUIRE(data.size() != 0);
  REQUIRE(data.empty() != true);
 }
