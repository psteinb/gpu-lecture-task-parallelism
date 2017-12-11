#include "catch.hpp"
#include "test_fixture.hpp"

#include "helper_cuda.h"

#include <iostream>
#include <chrono>
#include <vector>

__global__ void increment_kernel(int *g_data, int inc_value)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  g_data[idx] = g_data[idx] + inc_value + 10e6*(int)logf(sinf((float)idx) + tanf((float)idx));
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

std::size_t count_up(std::int32_t* data,
                             std::size_t size){

  int nbytes = size * sizeof(std::int32_t);
  int value = 1;

  int *h_a=0;
  checkCudaErrors(cudaHostAlloc((void **)&h_a, nbytes, cudaHostAllocPortable));
  std::fill(h_a,h_a + size,42);
  std::copy(data,data+size,h_a);

  // allocate device memory
  int *d_a=0;
  checkCudaErrors(cudaMalloc((void **)&d_a, nbytes));
  checkCudaErrors(cudaMemset(d_a, 255, nbytes));

  // set kernel launch configuration
  dim3 threads = dim3(512, 1);
  dim3 blocks  = dim3(size / threads.x, 1);
  unsigned long int counter=0;

  cudaEvent_t kstart, kend;
  cudaEventCreate (&kstart);
  cudaEventCreate (&kend);

  cudaEventRecord(kstart, 0);
  cudaMemcpyAsync(d_a, h_a, nbytes, cudaMemcpyHostToDevice,0);
  increment_kernel<<<blocks, threads,0,0>>>(d_a, value);
  cudaMemcpyAsync(data, d_a, nbytes, cudaMemcpyDeviceToHost,0);
  cudaEventRecord(kend, 0);

  while (cudaEventQuery(kend) == cudaErrorNotReady)
  {
    counter++;
  }

  checkCudaErrors(cudaFree(d_a));

  return counter;
}



TEST_CASE_METHOD(array_fixture, "simple_cuda_increment_works" ) {

  auto c = count_up(ints.data(), ints.size());
  
  REQUIRE(c != 0);
  REQUIRE(c > 10e6);
  
}
