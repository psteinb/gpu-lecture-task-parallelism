#ifndef HISTOGRAM_FIXTURE_H
#define HISTOGRAM_FIXTURE_H

#include <vector>
#include <cstdint>
#include <random>
#include <cmath>

template <int n_mega_elements = 16>
struct histogram_fixture {

  int frame_size;
  std::vector<std::uint8_t> histo;

  array_fixture() :
    frame_size(1024*1024),
    histo(n_mega_elements*1024*1024,0){

    std::random_device rd{20171212};
    std::mt19937 gen{rd()};

    std::normal_distribution<> d{64,20};

    for( std::uint8_t& el : histo )
      el = std::round(d(gen));
  }

};

#endif /* HISTOGRAM_FIXTURE_H */
