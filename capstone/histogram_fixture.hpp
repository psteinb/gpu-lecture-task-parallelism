#ifndef HISTOGRAM_FIXTURE_H
#define HISTOGRAM_FIXTURE_H

#include <vector>
#include <cstdint>
#include <random>
#include <cmath>

template <int n_mega_elements = 16>
struct histogram_fixture {

  int frame_size;
  std::vector<std::uint8_t> image;
  std::vector<std::uint32_t> histo;

  histogram_fixture() :
    frame_size(1024*1024),
    image(n_mega_elements*1024*1024,0),
    histo(256,0){

    //    std::random_device rd{20171212};
    std::mt19937 gen{20171212};

    std::normal_distribution<> d{64,20};

    for( std::uint8_t& el : image )
      el = std::round(d(gen));
  }

};

#endif /* HISTOGRAM_FIXTURE_H */
