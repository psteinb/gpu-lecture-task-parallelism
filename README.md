# gpu-lecture-task-parallelism

Lecture given at TU Dresden on task parallelism with GPUs (CUDA, OpenACC).

## Working on Taurus (if needed)

First, establish an interactive bash session on a gpu host (attention: the reservation flag will only work during class hours, omit them anytime else):

  + for Dec 10, 2018
  ```
  $ srun --reservation StreamsCUDA1 -A p_lv_gpu -t 1:30:00 --mem=4000 --gres=gpu:1 --partition=gpu2-interactive --pty bash -l
  ``` 
  
  + for Dec 11, 2018
  ```
  $ srun --reservation StreamsCUDA2 -A p_lv_gpu -t 1:30:00 --mem=4000 --gres=gpu:1 --partition=gpu2-interactive --pty bash -l
  ```

Second, please setup the correct environment (the defaul CUDA on taurus is version `9.2`):

```
$ module add modenv/scs5

Module GCCcore/6.4.0, zlib/1.2.11-GCCcore-6.4.0, cURL/7.58.0-GCCcore-6.4.0, expat/2.2.5-GCCcore-6.4.0, XZ/5.2.3-GCCcore-6.4.0, libxml2/2.9.4-GCCcore-6.4.0, ncurses/6.0-GCCcore-6.4.0, gettext/0.19.8.1-GCCcore-6.4.0, Perl/5.26.1-GCCcore-6.4.0, git/2.18.0-GCCcore-6.4.0 unloaded.
Module GCCcore/6.4.0, zlib/1.2.11-GCCcore-6.4.0, cURL/7.58.0-GCCcore-6.4.0, expat/2.2.5-GCCcore-6.4.0, XZ/5.2.3-GCCcore-6.4.0, libxml2/2.9.4-GCCcore-6.4.0, ncurses/6.0-GCCcore-6.4.0, gettext/0.19.8.1-GCCcore-6.4.0, Perl/5.26.1-GCCcore-6.4.0, git/2.18.0-GCCcore-6.4.0 loaded.
$ module load CUDA
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


   
