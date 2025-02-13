# GPTER

Roughly following Karpathy's videos on building a GPT with Tinygrad to train on my own datasets.


## notes
+ So far not able to match pytorch speed running [ng-video-lecture](https://github.com/karpathy/ng-video-lecture/) example code. _needs stats_
+ running with `CUDA=1 python gpter.py` seems to work ok. beam search :(
+ beam search is borked: `AssertionError: can only open device CUDA from parent, not SpawnPoolWorker-XX`. breaks on the first case of trying to realize a tensor.

### current drivers:
```
[ NVIDIA-SMI 570.86.15              Driver Version: 570.86.15      CUDA Version: 12.8 ]

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Jun__6_02:18:23_PDT_2024
Cuda compilation tools, release 12.5, V12.5.82
Build cuda_12.5.r12.5/compiler.34385749_0
```

## test run results
> _fuckin overwrote the file on accident_
