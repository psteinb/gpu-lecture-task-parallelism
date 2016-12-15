#ifndef TEST_FIXTURE_H
#define TEST_FIXTURE_H

#include <vector>
#include <cstdint>

#ifndef ARRAY_SIZE
static const std::size_t n_elements = 16* 1024*1024;
#else
static const std::size_t n_elements = ARRAY_SIZE;
#endif

struct array_fixture {

  std::vector<float> floats;
  std::vector<std::int32_t> ints;

  array_fixture() :
    floats(n_elements,0),
    ints(n_elements,0){

    std::size_t counter = 0;
    
    for( float& el : floats ){
      el = counter;
      ints[counter] = counter;
      ++counter;
    }

  }
  
};

#endif /* TEST_FIXTURE_H */
