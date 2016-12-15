# gpu-lecture-task-parallelism

Lecture given at TU Dresden on task parallelism with GPUs (CUDA, OpenACC).

## Working on Taurus

First, establish an interactive bash session on a gpu host:

```
$ srun --reservation=p_lv_cudaopencl_xx --pty --partition=gpu2-interactive -n 1 -c 1 --time=1:30:00 --mem-per-cpu=1700 --gres=gpu:1 bash
```

Second, please setup the correct environment (the defaul CUDA on taurus is version `8.0.44` which works at most with gcc `5.3.0`):

```
$ module load cuda gcc/5.3.0 
```

## Working with the code

Note that the examples use C++11. please make sure that the installed host side compiler supports this standard.

To start working, change your working directory to where your code lives and checkout the repo :

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
