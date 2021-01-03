#include <chrono>
#include <iostream>

#include "catch.hpp"
#include "test_fixture.hpp"

#include "helper_cuda.h"


__global__ void increment_kernel(int *g_data, int inc_value)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  g_data[idx] = g_data[idx] + inc_value;
}

double increment_by_two(std::int32_t* data,
		      std::size_t size)
{

  const int nbytes = size * sizeof(std::int32_t);
  const int value = 2;

  // allocate device memory
  int *d_a=0;
  checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));
  checkCudaErrors(cudaMemset(d_a, 0, nbytes));

  // set kernel launch configuration
  dim3 threads = dim3(512, 1);
  dim3 blocks  = dim3(size / threads.x, 1);

  auto start = std::chrono::high_resolution_clock::now();

  cudaMemcpy(d_a, data, nbytes, cudaMemcpyHostToDevice);
  increment_kernel<<<blocks, threads>>>(d_a, value);
  cudaMemcpy(data, d_a, nbytes, cudaMemcpyDeviceToHost);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end-start;

  checkCudaErrors(cudaFree(d_a));

  return diff.count();
}

#include "omp.h"
double omp_increment_by_two(std::int32_t* data,
				 std::size_t size)
{

  const int nbytes = size * sizeof(std::int32_t);
  const int value = 1;

  // allocate device memory
  int *d_a=0;
  checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));
  checkCudaErrors(cudaMemset(d_a, 0, nbytes));

  // set kernel launch configuration
  dim3 threads = dim3(512, 1);
  dim3 blocks  = dim3(size / threads.x, 1);
  
  auto start = std::chrono::high_resolution_clock::now();
  cudaMemcpy(d_a, data, nbytes, cudaMemcpyHostToDevice);

  omp_set_num_threads(2); 
  #pragma omp parallel
  {
  increment_kernel<<<blocks, threads>>>(d_a, value);
  }

  cudaMemcpy(data, d_a, nbytes, cudaMemcpyDeviceToHost);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end-start;
  
  checkCudaErrors(cudaFree(d_a));
  return diff.count();
}


TEST_CASE_METHOD(array_fixture, "increment_works" ) {

  const int value0 = ints[0];
  REQUIRE(ints[0]==0);
  
  double timing = increment_by_two(ints.data(),ints.size());

  REQUIRE(timing > 0.);
  
    
  REQUIRE(value0 != ints[0]);
  REQUIRE(value0 + 2 == ints[0]);
 
}

TEST_CASE_METHOD(array_fixture, "omp_increment_works" ) {

  const int value0 = ints[0];
  REQUIRE(ints[0]==0);
  
  double timing = omp_increment_by_two(ints.data(),ints.size());

  REQUIRE(timing > 0.);
      
  REQUIRE(value0 != ints[0]);
  REQUIRE(value0 + 2 == ints[0]);
 
}

TEST_CASE_METHOD(array_fixture, "omp_faster" ) {

  //warm-up
  increment_by_two(ints.data(),ints.size());
  
  double omp_timing = omp_increment_by_two(ints.data(),ints.size());
  double ser_timing = increment_by_two(ints.data(),ints.size());

  REQUIRE(omp_timing < ser_timing);
       
}

