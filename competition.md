# Competition on Task Parallelism

## Preface

We are asked to produce a histogram on a 8-bit integer 2D image. The images vary in size and can be 16 MB, 64 MB and 128 MB. The images are provided by a pseudo-random number generator. that produces a mean of 42 and a standard deviation of 10.

A histogram for 8-bit pixel values can be computed like so:

```

std::vector<std::size_t> histo(256,0); // can also be a stack allocated buffer etc

for( std::uint8_t& pixel : given_image ){
	histo[pixel]+=1;
}

```

## Task

Write a CUDA implementation of computing a histogram for 8-bit 2D images of the above given size(s).  Exploit Cuda Streams to overlap computation and PCIe memory transfers as much as possible.

Perform 1 warm-up run and 20 benchmark runs of your implementation. Measure the runtime of every run of those 20 samples and compute at least the arithmetic mean and min/max (at best median and the standard deviation around the median) of the runtimes to allow a fair comparison. 

### Beat the CPU!

Compare your application to a pure OpenMP implementation. Explain the difference in runtime!

### Beat yourself?

Does you implementation scale (linearly) with the size of the input image? Use `nvprof` to analyse your implementation. 

## Bonus

Measure the pure computation time on the GPU. Compare the scaling of the device time to the total time to solution. Why does it differ from the time to solution from above?
