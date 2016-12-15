# gpu-lecture-task-parallelism

Lecture given at TU Dresden on task parallelism with GPUs (CUDA, OpenACC).

## Working on taurus

## running the code

### Working on Taurus

First, establish an interactive bash session on a gpu host:

```
$ srun --reservation=p_lv_cudaopencl_xx --pty --partition=gpu2-interactive -n 1 -c 1 --time=1:30:00 --mem-per-cpu=1700 bash
```

Second, please setup the correct environment:

```
$ module load cuda
```

Then, change your working directory to where your code lives and checkout the repo :

```
$ cd /path/to/where/I/want/to/work
$ git clone https://github.com/psteinb/gpu-lecture-task-parallelism.git
$ cd gpu-lecture-task-parallelism
```
