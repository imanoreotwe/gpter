# GPTER

Roughly following Karpathy's videos on building a GPT with Tinygrad to train on my own datasets.


## notes
+ benchmark `Tensor.randint()` vs `np.random.randint()`
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

### ng-video-lecture stats:
```
49.386577 M parameters
step 0, train loss: 10.9164, training time: nans, training speed: nan tok/s
step 500, train loss: 5.6129, training time: 0.0965s, training speed: 170053.9844 tok/s
step 1000, train loss: 5.0071, training time: 0.0969s, training speed: 169090.0938 tok/s
step 1500, train loss: 4.6465, training time: 0.0969s, training speed: 169050.4062 tok/s
step 2000, train loss: 4.4359, training time: 0.0969s, training speed: 169083.7500 tok/s
step 2500, train loss: 4.2596, training time: 0.0971s, training speed: 168816.6719 tok/s
step 3000, train loss: 4.1371, training time: 0.0970s, training speed: 168840.2188 tok/s
step 3500, train loss: 4.0268, training time: 0.0972s, training speed: 168552.7812 tok/s
step 4000, train loss: 3.9499, training time: 0.0971s, training speed: 168676.5781 tok/s
step 4500, train loss: 3.8726, training time: 0.0973s, training speed: 168470.2969 tok/s
step 4999, train loss: 3.8037, training time: 0.0974s, training speed: 168228.1719 tok/s
```
I'm suspicious of that training speed...
