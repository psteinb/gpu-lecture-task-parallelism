#include "catch.hpp"
#include "histogram_fixture.hpp"

#include "helper_cuda.h"

#include <iterator>

TEST_CASE_METHOD(histogram_fixture<16> , "fixture_works" ) {
  REQUIRE(histo.size() != 0);
  REQUIRE(histo.empty() != true);

}


TEST_CASE_METHOD(histogram_fixture<16> , "histogram_right" ) {

  auto max_el = std::max_element(histo.cbegin(),histo.cend());

  REQUIRE( max_el != histo.cbegin());
}
