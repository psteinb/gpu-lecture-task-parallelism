#include "catch.hpp"
#include "competition_fixture.hpp"

#include "helper_cuda.h"


TEST_CASE_METHOD(competition_fixture, "fixture_works" ) {
  REQUIRE(small.size() != 0);
  REQUIRE(medium.size() != 0);
  REQUIRE(large.size() != 0);
 
}

TEST_CASE_METHOD(competition_fixture, "mean_is_42" ) {

  REQUIRE(std::abs(mean(small.begin(), small.end()) - 42.) < 1);
  REQUIRE(std::abs(mean(medium.begin(), medium.end()) - 42.) < 1);
  REQUIRE(std::abs(mean(large.begin(), large.end()) - 42.) < 1);
   
}

TEST_CASE_METHOD(competition_fixture, "sd_is_around_10" ) {

  REQUIRE(std::abs(sd(small.begin(), small.end()) - 10.) < 1);
  REQUIRE(std::abs(sd(medium.begin(), medium.end()) - 10.) < 1);
  REQUIRE(std::abs(sd(large.begin(), large.end()) - 10.) < 1);
   
}



