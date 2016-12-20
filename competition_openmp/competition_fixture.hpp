#ifndef COMPETE_FIXTURE_H
#define COMPETE_FIXTURE_H

#include <vector>
#include <cstdint>

struct competition_fixture {

  std::vector<std::int8_t> small;
  std::vector<std::int8_t> medium;
  std::vector<std::int8_t> large;
  
  competition_fixture() :
    small(4*1024*1024,42),
    medium(16*1024*1024,42),
    large(32*1024*1024,42)
  {

  }
  
};

#endif /* TEST_FIXTURE_H */
