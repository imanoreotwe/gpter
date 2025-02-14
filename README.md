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
```
n_embed = 384
batch_size = 64
block_size = 256
learning_rate = 3e-4

39.253841 M parameters
step 0, mean loss: 10.8287, training time: 5.8788s, training speed: 2786.9497 tok/s
step 500, mean loss: 7.4588, training time: 1.8052s, training speed: 9077.5498 tok/s
step 1000, mean loss: 7.3541, training time: 1.8094s, training speed: 9056.5459 tok/s
step 1500, mean loss: 6.9720, training time: 1.8061s, training speed: 9072.6494 tok/s
step 2000, mean loss: 6.6607, training time: 1.8083s, training speed: 9062.6650 tok/s
step 2500, mean loss: 6.4127, training time: 1.8094s, training speed: 9057.5732 tok/s
step 3000, mean loss: 6.2090, training time: 1.8111s, training speed: 9049.5771 tok/s
step 3500, mean loss: 6.0680, training time: 1.8108s, training speed: 9047.8320 tok/s
step 4000, mean loss: 5.9530, training time: 1.8117s, training speed: 9047.8506 tok/s
step 4500, mean loss: 5.8494, training time: 1.8100s, training speed: 9051.7617 tok/s
step 5000, mean loss: 5.7834, training time: 1.8113s, training speed: 9045.5537 tok/s
step 5500, mean loss: 5.7173, training time: 1.8119s, training speed: 9042.5820 tok/s
step 6000, mean loss: 5.6505, training time: 1.8149s, training speed: 9034.3623 tok/s
step 6500, mean loss: 5.5871, training time: 1.8121s, training speed: 9041.6641 tok/s
step 7000, mean loss: 5.5250, training time: 1.8153s, training speed: 9035.1426 tok/s
step 7500, mean loss: 5.4661, training time: 1.8124s, training speed: 9039.9824 tok/s
step 8000, mean loss: 5.4323, training time: 1.8135s, training speed: 9034.7061 tok/s
step 8500, mean loss: 5.3950, training time: 1.8135s, training speed: 9034.8926 tok/s
step 9000, mean loss: 5.3498, training time: 1.8135s, training speed: 9035.0059 tok/s
step 9500, mean loss: 5.2947, training time: 1.8143s, training speed: 9030.6777 tok/s
EVAL LOSS: 5.3527
```

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
