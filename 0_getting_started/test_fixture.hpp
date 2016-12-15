#ifndef TEST_FIXTURE_H
#define TEST_FIXTURE_H

#include <vector>

#ifndef ARRAY_SIZE
static const std::size_t n_elements = 16* 1024*1024;
#else
static const std::size_t n_elements = ARRAY_SIZE;
#endif

struct array_fixture {

  std::vector<float> data;

  array_fixture() :
    data(n_elements,0){

    std::size_t counter = 0;
    for( float& el : data )
      el = counter++;
    
  }
  
};

#endif /* TEST_FIXTURE_H */
