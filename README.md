# gpu-lecture-task-parallelism

Lecture given at TU Dresden on task parallelism with GPUs (CUDA, OpenACC).

## Working on Taurus (if needed)

First, establish an interactive bash session on a gpu host (attention: the reservation flag will only work during class hours, omit them anytime else):

  + for Dec 11, 2017 
  ```
  $ srun --reservation p_lv_cudaopencl_248 -A p_lv_cudaopencl -t 90 --mem=4000 --gres=gpu:1 --partition=gpu2-interactive --pty bash -l
  ``` 
  
  + for Dec 12, 2017
  ```
  $ srun --reservation p_lv_cudaopencl_249 -A p_lv_cudaopencl -t 90 --mem=4000 --gres=gpu:1 --partition=gpu2-interactive --pty bash -l
  ```

Second, please setup the correct environment (the defaul CUDA on taurus is version `8.0.44` which works at most with gcc `5.3.0`):

```
$ module load cuda gcc/5.3.0 
```

## Working with the code

Note that the examples use C++11. please make sure that the installed host side compiler supports this standard. If your host uses gcc as the default c/c++ compiler, any version between 4.9 and 5.4 should be fine with CUDA 8.

To prepare for class, change your working directory to where your code lives and checkout this repo :

```
$ cd /path/to/where/I/want/to/work
$ git clone https://github.com/psteinb/gpu-lecture-task-parallelism.git
$ cd gpu-lecture-task-parallelism
```

Let's start with the acceptance test:

```
$ cd 0_getting_started/
$ make
$ ./test_simple_increment
```

If this test passes, you are ready to go, if not, diagnose the problem. Feel free to post an issue to this repo if you are unable to solve the problem.


   
