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

  std::vector<cudaStream_t> streams(2);
  for( cudaStream_t& el : streams ){
    cudaStreamCreate(&el);
  }

  // allocate device memory
  int * h_a[2];
  for(int s = 0;s < streams.size();++s){
    checkCudaErrors(cudaHostAlloc((void **)&h_a[s], nbytes/2, cudaHostAllocPortable));
    std::copy(data+s*(size/2), data+(s+1)*(size/2),h_a[s]);
  }

  // allocate device memory
  int *d_a[2];
  for(int s = 0;s < streams.size();++s){
    checkCudaErrors(cudaMalloc((void **)&d_a[s], nbytes));
    checkCudaErrors(cudaMemset(d_a[s], 255, nbytes));
  }

  // set kernel launch configuration
  dim3 threads = dim3(512, 1);
  dim3 blocks  = dim3(size / (2*threads.x), 1);


  auto start = std::chrono::high_resolution_clock::now();

  for(int s = 0;s < streams.size();++s){
    cudaMemcpyAsync(d_a[s], h_a[s], nbytes/2, cudaMemcpyHostToDevice, streams[s]);
    increment_kernel<<<blocks, threads,0,streams[s]>>>(d_a[s], value);
    cudaMemcpyAsync( h_a[s],d_a[s], nbytes/2, cudaMemcpyDeviceToHost, streams[s]);
  }

  cudaDeviceSynchronize();

  auto end = std::chrono::high_resolution_clock::now();

  for(int s = 0;s < streams.size();++s){
    checkCudaErrors(cudaFree(d_a[s]));
    std::copy(h_a[s],h_a[s]+size/2,data+s*(size/2));
    checkCudaErrors(cudaFreeHost(h_a[s]));
  }

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
  REQUIRE(ints[0] == 1);

}

TEST_CASE_METHOD(array_fixture, "compare_times" ) {

  auto serial = increment_by_one(ints.data(), ints.size());
  auto parallel = parallel_increment_by_one(ints.data(), ints.size());

  REQUIRE(ints[0] != 0);
  REQUIRE(ints[0] == 2);

  REQUIRE(parallel < serial);
}
