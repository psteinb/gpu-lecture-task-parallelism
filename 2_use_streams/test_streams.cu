#include "catch.hpp"
#include "test_fixture.hpp"

#include "helper_cuda.h"

#include <iostream>
#include <chrono>
#include <vector>

__global__ void increment_kernel(int *g_data, int inc_value)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  g_data[idx] = g_data[idx] + inc_value;
}

double increment_by_one(std::int32_t* data,
		      std::size_t size)
{

  int nbytes = size * sizeof(std::int32_t);
  int value = 1;

  // allocate device memory
  int *d_a=0;
  checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));
  checkCudaErrors(cudaMemset(d_a, 255, nbytes));

  // set kernel launch configuration
  dim3 threads = dim3(512, 1);
  dim3 blocks  = dim3(size / threads.x, 1);


  auto start = std::chrono::high_resolution_clock::now();

  cudaMemcpy(d_a, data, nbytes, cudaMemcpyHostToDevice);
  increment_kernel<<<blocks, threads>>>(d_a, value);
  cudaMemcpy(data, d_a, nbytes, cudaMemcpyDeviceToHost);

  auto end = std::chrono::high_resolution_clock::now();

  checkCudaErrors(cudaFree(d_a));

  return (end - start).count();
}

double parallel_increment_by_one(std::int32_t* data,
                             std::size_t size)
{

  int nbytes = size * sizeof(std::int32_t);
  int value = 1;

  // allocate device memory
  int *d_a=0;
  checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));
  checkCudaErrors(cudaMemset(d_a, 255, nbytes));

  // set kernel launch configuration
  dim3 threads = dim3(512, 1);
  dim3 blocks  = dim3(size / threads.x, 1);

  std::vector<cudaStream_t> streams(4);
  for( cudaStream_t& el : streams ){
    cudaStreamCreate(&el);
  }

  auto start = std::chrono::high_resolution_clock::now();

  cudaMemcpy(d_a, data, nbytes, cudaMemcpyHostToDevice);

  for(int s = 0;s < streams.size();++s){
    std::cout << "increment_kernel " << s << "/" << streams.size() << "\n";
    increment_kernel<<<blocks, threads,0,streams[s]>>>(d_a, value);
  }

  cudaMemcpy(data, d_a, nbytes, cudaMemcpyDeviceToHost);
  auto end = std::chrono::high_resolution_clock::now();

  checkCudaErrors(cudaFree(d_a));

  return (end - start).count();
}


TEST_CASE_METHOD(array_fixture, "simple_cuda_increment_works" ) {

  increment_by_one(ints.data(), ints.size());
  
  REQUIRE(ints[0] != 0);
  REQUIRE(ints[0] == 1);
  
}

TEST_CASE_METHOD(array_fixture, "streams_increment_works" ) {

  parallel_increment_by_one(ints.data(), ints.size());

  REQUIRE(ints[0] != 0);
  REQUIRE(ints[0] == 4);

}

TEST_CASE_METHOD(array_fixture, "compare_times" ) {

  auto serial = increment_by_one(ints.data(), ints.size());
  auto parallel = parallel_increment_by_one(ints.data(), ints.size());

  REQUIRE(ints[0] != 0);
  REQUIRE(ints[0] == 5);

  REQUIRE(parallel < serial);
}