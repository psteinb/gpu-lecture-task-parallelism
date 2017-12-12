#include "catch.hpp"
#include "histogram_fixture.hpp"

#include "helper_cuda.h"


TEST_CASE_METHOD(histogram_fixture<16> , "fixture_works" ) {
  REQUIRE(histo.size() != 0);
  REQUIRE(histo.empty() != true);

}
