# swiGLU PTO-ISA vs torch_npu

## msprof on simulator
|PTO-ISA|torch_npu|
|---------|-----------|
|![PTO-ISA](https://github.com/Mocchibird/pto-kernels-plots/blob/main/swiglu/swiglu_pto-isa.png?raw=true)|![torch_npu](https://github.com/Mocchibird/pto-kernels-plots/blob/main/swiglu/swiglu_torch.png?raw=true)|

## msprof on device
|PTO-ISA|torch_npu|
|---------|-----------|
|![PTO-ISA](https://github.com/Mocchibird/pto-kernels-plots/blob/main/swiglu/swiglu_pto-isa_on_device.png?raw=true)|![torch_npu](https://github.com/Mocchibird/pto-kernels-plots/blob/main/swiglu/swiglu_torch_on_device.png?raw=true)|

PTO-ISA is around `24% faster` than torch_npu for simulated profiling and `13% faster`for on device profiling.

## python benchmark
|Speedup Heatmap|
|-----------|
|![Speedup Heatmap](https://github.com/Mocchibird/pto-kernels-plots/blob/main/swiglu/swiglu_speedup_heatmap_bd24.png?raw=true)|

But in the python benchmark it is has `0.78x` the speed of torch_npu.
This is because of launch overhead, because for larger sizes for example (16384, 16384)
|Type|PTO_ISA|torch_npu|
|---|---|---|
|Total Time|1352.691us|1382.863us|
|Kernel Time|1352.307us|1382.588us|
