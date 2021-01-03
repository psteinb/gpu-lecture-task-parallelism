#include <chrono>
#include <iostream>

#include "catch.hpp"
#include "test_fixture.hpp"

#include "helper_cuda.h"


__global__ void increment_kernel(int *g_data, int inc_value, int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int dim = blockDim.x * gridDim.x;

  for(; idx < size; idx += dim)
      g_data[idx] = g_data[idx] + inc_value;
}

double increment_by_two(std::int32_t* data,
		      std::size_t size)
{

  const int nbytes = size * sizeof(std::int32_t);
  const int value = 8;

  // allocate device memory
  int *d_a=0;
  checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));
  checkCudaErrors(cudaMemset(d_a, 0, nbytes));

  // set kernel launch configuration
  dim3 threads = dim3(512, 1);
  dim3 blocks  = dim3(4*8, 1);

  auto start = std::chrono::high_resolution_clock::now();

  cudaMemcpy(d_a, data, nbytes, cudaMemcpyHostToDevice);
  increment_kernel<<<blocks, threads>>>(d_a, value, size /*threads.x*blocks.x*/);
  cudaMemcpy(data, d_a, nbytes, cudaMemcpyDeviceToHost);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end-start;

  checkCudaErrors(cudaFree(d_a));

  return diff.count();
}

//#include "omp.h"
double streamed_increment_by_two(std::int32_t* data,
				 std::size_t size)
{

  const int nbytes = size * sizeof(std::int32_t);
  const int value = 8;
  //const std::size_t half = size/2;
  
  // allocate device memory
  int *d_a=0;
  checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));
  checkCudaErrors(cudaMemset(d_a, 0, nbytes));

  // set kernel launch configuration
  dim3 threads = dim3(512, 1);
  dim3 blocks  = dim3(4, 1);
  std::cout << "threads = "<< threads.x <<", blocks = " << blocks.x << " of " << size << " elements\n";
  
  auto start = std::chrono::high_resolution_clock::now();
  cudaMemcpy(d_a, data, nbytes, cudaMemcpyHostToDevice);

  std::vector<cudaStream_t> streams(8);
  const std::size_t subSize = size/streams.size() /*threads.x*blocks.x*/;
  for(cudaStream_t& stream : streams)
    cudaStreamCreate(&stream);

  for(int i = 0;i<streams.size();i++)
    increment_kernel<<<blocks, threads,0,streams[i]>>>(d_a + i*subSize, value, subSize);

  cudaMemcpy(data, d_a, nbytes, cudaMemcpyDeviceToHost);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end-start;

  checkCudaErrors(cudaFree(d_a));
  for(cudaStream_t& stream : streams)
    cudaStreamDestroy(stream);

  return diff.count();
}


TEST_CASE_METHOD(array_fixture, "increment_works" ) {

  const int value0 = ints[0];
  REQUIRE(ints[0]==0);
  
  double timing = increment_by_two(ints.data(),ints.size());

  REQUIRE(timing > 0.);
  
    
  REQUIRE(value0 != ints[0]);
  REQUIRE(value0 + 8 == ints[0]);
 
}

TEST_CASE_METHOD(array_fixture, "streamed_increment_works" ) {

  const int value0 = ints[0];
  REQUIRE(ints[0]==0);
  
  double timing = streamed_increment_by_two(ints.data(),ints.size());

  REQUIRE(timing > 0.);
      
  REQUIRE(value0 != ints[0]);
  REQUIRE(value0 + 8 == ints[0]);
 
}

TEST_CASE_METHOD(array_fixture, "streamed_faster" ) {

  //warm-up
  increment_by_two(ints.data(),ints.size());
  
  double streamed_timing = streamed_increment_by_two(ints.data(),ints.size());
  double ser_timing = increment_by_two(ints.data(),ints.size());
  std::cout << "streamed_timing = " << streamed_timing << ", serial = " << ser_timing << "\n";
    
  REQUIRE(streamed_timing < ser_timing);
       
}

