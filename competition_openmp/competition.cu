#include "catch.hpp"
#include "competition_fixture.hpp"

#include "helper_cuda.h"


TEST_CASE_METHOD(competition_fixture, "fixture_works" ) {
  REQUIRE(small.size() != 0);
  REQUIRE(medium.size() != 0);
  REQUIRE(large.size() != 0);
 
}

