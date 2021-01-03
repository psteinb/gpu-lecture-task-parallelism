#include <iostream>
#include <cmath>

#include "catch.hpp"
#include "test_fixture.hpp"

#include "helper_cuda.h"

__global__ void increment_kernel(int *g_data, int inc_value)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  g_data[idx] = g_data[idx] + inc_value;
}

__global__ void add_kernel(int *g_A, int* g_B)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  g_A[idx] += g_B[idx];
}


void increment_by_one(std::int32_t* data,
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

  cudaMemcpy(d_a, data, nbytes, cudaMemcpyHostToDevice);
  increment_kernel<<<blocks, threads>>>(d_a, value);
  cudaMemcpy(data, d_a, nbytes, cudaMemcpyDeviceToHost);

  checkCudaErrors(cudaFree(d_a));
}

float async_increment_by_one(std::int32_t* data,
			    std::size_t size)
{

  int nbytes = size * sizeof(std::int32_t);
  int value = 1;
  const std::size_t half = size/2;
  
  // allocate device memory
  int *d_a[2] ={nullptr,nullptr};
  int *d_b[2] ={nullptr,nullptr};
  int *h_a[2] ={nullptr,nullptr};
  int *h_b[2] ={nullptr,nullptr};
  cudaStream_t streams[4];
  
  checkCudaErrors(cudaMallocHost(&h_a[0], nbytes/2));
  checkCudaErrors(cudaMallocHost(&h_a[1], nbytes/2));
  std::fill(h_a[0], h_a[0] + half, 0);
  std::fill(h_a[1], h_a[1] + half, 0);

  checkCudaErrors(cudaMallocHost(&h_b[0], nbytes/2));
  checkCudaErrors(cudaMallocHost(&h_b[1], nbytes/2));
  std::copy(data,data+half,h_b[0]);
  std::copy(data+half,data+size,h_b[1]);
  
  checkCudaErrors(cudaMalloc((void **)&d_a[0], nbytes/2));
  checkCudaErrors(cudaMalloc((void **)&d_a[1], nbytes/2));
  checkCudaErrors(cudaMalloc((void **)&d_b[0], nbytes/2));
  checkCudaErrors(cudaMalloc((void **)&d_b[1], nbytes/2));

  for(auto& a : streams)
    checkCudaErrors(cudaStreamCreate(&a));
  
  // set kernel launch configuration
  dim3 threads = dim3(512, 1);
  dim3 blocks  = dim3(half / (threads.x), 1);
  cudaEvent_t kernel1_start, kernel1_end, kernel2_end;
  cudaEventCreate (&kernel1_start); 
  cudaEventCreate (&kernel1_end);
  cudaEventCreate (&kernel2_end);
 
  cudaMemcpyAsync(d_a[0], h_a[0], nbytes/2, cudaMemcpyHostToDevice,streams[0]);
  cudaMemcpyAsync(d_a[1], h_a[1], nbytes/2, cudaMemcpyHostToDevice,streams[1]);

  // having this event delays kernel1 until the HtoD memcpy finished for some reason
  cudaEventRecord (kernel1_start, streams[0]);
  increment_kernel<<<blocks, threads,0,streams[0]>>>(d_a[0], value);
  increment_kernel<<<blocks, threads,0,streams[1]>>>(d_a[1], value);
  cudaEventRecord (kernel1_end, streams[0]);
  cudaEventRecord (kernel2_end, streams[1]);

  cudaMemcpyAsync(d_b[0], h_b[0], nbytes/2, cudaMemcpyHostToDevice,streams[0+2]);
  cudaMemcpyAsync(d_b[1], h_b[1], nbytes/2, cudaMemcpyHostToDevice,streams[1+2]);

  cudaStreamWaitEvent(streams[0+2], kernel1_end, 0);
  cudaStreamWaitEvent(streams[1+2], kernel2_end, 0);

  add_kernel<<<blocks, threads,0,streams[0+2]>>>(d_b[0], d_a[0]);
  add_kernel<<<blocks, threads,0,streams[1+2]>>>(d_b[1], d_a[1]);

  cudaMemcpyAsync(h_a[0], d_b[0], nbytes/2, cudaMemcpyDeviceToHost,streams[0+2]);
  cudaMemcpyAsync(h_a[1], d_b[1], nbytes/2, cudaMemcpyDeviceToHost,streams[1+2]);

  cudaDeviceSynchronize();
  float time = 1;
  cudaEventElapsedTime(&time,kernel1_start,kernel1_end);
  
  std::copy(h_a[0],h_a[0]+half,data);
  std::copy(h_a[1],h_a[1]+half,data+half);
  
  checkCudaErrors(cudaFree(d_a[0]));
  checkCudaErrors(cudaFree(d_a[1]));
  checkCudaErrors(cudaFree(d_b[0]));
  checkCudaErrors(cudaFree(d_b[1]));
  checkCudaErrors(cudaFreeHost(h_a[0]));
  checkCudaErrors(cudaFreeHost(h_a[1]));
  checkCudaErrors(cudaFreeHost(h_b[0]));
  checkCudaErrors(cudaFreeHost(h_b[1]));
  for(auto& a : streams)
    checkCudaErrors(cudaStreamDestroy(a));

  return time;
}



TEST_CASE_METHOD(array_fixture, "async_cuda_increment_works" ) {

  async_increment_by_one(ints.data(), ints.size());
  
  REQUIRE(ints[0] != 0);
  REQUIRE(ints[0] == 1);
  
}

TEST_CASE_METHOD(array_fixture, "async_same_as_normal" ) {

  auto ints2 = ints;

  increment_by_one(ints2.data(), ints2.size());
  async_increment_by_one(ints.data(), ints.size());
    
  REQUIRE(ints[0] == ints2[0]);
  
}


TEST_CASE_METHOD(array_fixture, "time_as_expected" ) {

  float time_ms = async_increment_by_one(ints.data(), ints.size());
  std::cout << "time = " << time_ms << " ms\n";
  REQUIRE(time_ms != 0.);
  REQUIRE(time_ms < 1000.);
}
