#include "catch.hpp"
#include "test_fixture.hpp"

#include "helper_cuda.h"

__global__ void increment_kernel(int *g_data, int inc_value)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  g_data[idx] = g_data[idx] + inc_value;
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

void async_increment_by_one(std::int32_t* data,
			    std::size_t size)
{

  int nbytes = size * sizeof(std::int32_t);
  int value = 1;
  const std::size_t half = size/2;
  
  // allocate device memory
  int *d_a[2] ={nullptr,nullptr};
  int *h_a[2] ={nullptr,nullptr};
  cudaStream_t streams[2];
  
  checkCudaErrors(cudaMallocHost(&h_a[0], nbytes/2));
  checkCudaErrors(cudaMallocHost(&h_a[1], nbytes/2));
  std::copy(data,data+half,h_a[0]);
  std::copy(data+half,data+size,h_a[1]);
  
  checkCudaErrors(cudaMalloc((void **)&d_a[0], nbytes/2));
  checkCudaErrors(cudaMalloc((void **)&d_a[1], nbytes/2));

  checkCudaErrors(cudaStreamCreate(&streams[0]));
  checkCudaErrors(cudaStreamCreate(&streams[1]));
  
  // set kernel launch configuration
  dim3 threads = dim3(512, 1);
  dim3 blocks  = dim3(half / (threads.x), 1);

  cudaMemcpyAsync(d_a[0], h_a[0], nbytes/2, cudaMemcpyHostToDevice,streams[0]);
  cudaMemcpyAsync(d_a[1], h_a[1], nbytes/2, cudaMemcpyHostToDevice,streams[1]);
  increment_kernel<<<blocks, threads,0,streams[0]>>>(d_a[0], value);
  increment_kernel<<<blocks, threads,0,streams[1]>>>(d_a[1], value);
  cudaMemcpyAsync(h_a[0], d_a[0], nbytes/2, cudaMemcpyDeviceToHost,streams[0]);
  cudaMemcpyAsync(h_a[1], d_a[1], nbytes/2, cudaMemcpyDeviceToHost,streams[1]);

  cudaDeviceSynchronize();
  
  
  std::copy(h_a[0],h_a[0]+half,data);
  std::copy(h_a[1],h_a[1]+half,data+half);
  
  checkCudaErrors(cudaFree(d_a[0]));
  checkCudaErrors(cudaFree(d_a[1]));
  checkCudaErrors(cudaFreeHost(h_a[0]));
  checkCudaErrors(cudaFreeHost(h_a[1]));
  checkCudaErrors(cudaStreamDestroy(streams[0]));
  checkCudaErrors(cudaStreamDestroy(streams[1]));
  
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