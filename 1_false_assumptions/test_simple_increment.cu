#include <chrono>

#include "catch.hpp"
#include "test_fixture.hpp"

#include "helper_cuda.h"

__global__ void increment_kernel(int *g_data, int inc_value)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  g_data[idx] = g_data[idx] + inc_value;
}

double increment_by_one(std::int32_t* data,
		      std::size_t size)
{

  int nbytes = size * sizeof(std::int32_t);
  int value = 8;

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
  std::chrono::duration<double> diff = end-start;

  checkCudaErrors(cudaFree(d_a));

  return diff.count();
}

#include "omp.h"
double streamed_increment_by_one(std::int32_t* data,
			       std::size_t size)
{

  int nbytes = size * sizeof(std::int32_t);
  int value = 4;

  // allocate device memory
  int *d_a=0;
  checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));
  checkCudaErrors(cudaMemset(d_a, 255, nbytes));
  
  // set kernel launch configuration
  dim3 threads = dim3(512, 1);
  dim3 blocks  = dim3(size / threads.x, 1);
  
  auto start = std::chrono::high_resolution_clock::now();
  cudaMemcpy(d_a, data, nbytes, cudaMemcpyHostToDevice);

  omp_set_num_threads(2);
#pragma omp parallel
  {
    increment_kernel<<<blocks, threads,0,0>>>(d_a, value);
  }
  
  cudaMemcpy(data, d_a, nbytes, cudaMemcpyDeviceToHost);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end-start;

  checkCudaErrors(cudaFree(d_a));
  
  return diff.count();
}

TEST_CASE_METHOD(array_fixture, "fixture_works" ) {
  REQUIRE(ints.size() != 0);
  REQUIRE(ints.empty() != true);

  REQUIRE(floats.size() != 0);
  REQUIRE(floats.empty() != true);
 
}

TEST_CASE_METHOD(array_fixture, "simple_cuda_increment_works" ) {

  increment_by_one(ints.data(), ints.size());
  
  REQUIRE(ints[0] != 0);
  REQUIRE(ints[0] == 8);
  
}

TEST_CASE_METHOD(array_fixture, "streamed_cuda_increment_works" ) {

  streamed_increment_by_one(ints.data(), ints.size());
  
  REQUIRE(ints[0] != 0);
  REQUIRE(ints[0] == 8);
  
}

TEST_CASE_METHOD(array_fixture, "omp_faster" ) {

  double ser_timing = increment_by_one(ints.data(), ints.size());
  double omp_timing = streamed_increment_by_one(ints.data(), ints.size());
  
  REQUIRE(omp_timing < ser_timing);
  
}

