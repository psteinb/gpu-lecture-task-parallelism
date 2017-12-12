# Exercise on Task Parallelism

## Preface

We are asked to produce a histogram on a 8-bit integer 2D image. The images vary in size and can be 16 MB, 64 MB and 128 MB. The images are provided by a pseudo-random number generator. that produces a mean of 42 and a standard deviation of 10.

## Task

Write a CUDA implementation of computing a histogram for 8-bit 2D images of the above given size(s).  

A histogram is a collection of intensity frequencies which can be computed for any given ensemble of values. Say we have a 2D 8-bit image. 8-bit greyscale values can fall within the interval $[0,255]$. This means, that the histogram is nothing more than a integer array of size $256$ where each entry at offset $i$ corresponds to the number of occurrances of value $i$ in the image. An example implementation could be:

```
std::vector<std::uint8_t> my_image(32*32);
//my_image is full of values
std::vector<std::uint32_t> ahistogram(255,0);

for(std::size_t p = 0;p<my_image.size();++p){
    ahistogram[my_image[p]]+=1;
}
```

Exploit Cuda Streams to overlap computation and PCIe transfers as much as possible. Does you implementation scale linearly with the size of the input image?

## Bonus

Compare your application to a pure OpenMP implementation. Explain the difference in runtime!
