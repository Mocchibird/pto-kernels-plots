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
|Speedup jit-compiler/ctypes vs torch_npu|Speedup cmake/pybind vs torch_npu|
|-----------|----|
|![Speedup ctypes](https://github.com/Mocchibird/pto-kernels-plots/blob/main/swiglu/swiglu_speedup_heatmap_bd24.png?raw=true)|![Speedup pybind](https://github.com/Mocchibird/pto-kernels-plots/blob/main/swiglu/swiglu_speedup_heatmap_bd24_pybind.png?raw=true)|

The benchmark with pybind has `1.5x` the speed of torch_npu, while the benchmark with jit-compiler has `0.78x` the speed of torch_npu.
Most of the slowdown for the jit-compiler version comes from repeated stream_ptr lookup. Caching the stream_ptr speeds-up the jit-compiler version significantly for smaller shapes, but not much for larger shapes.

|Speedup jit-compiler/ctypes w. cached stream_ptr vs torch_npu|
|---|
|![Speedup ctypes cached stream_ptr](https://github.com/Mocchibird/pto-kernels-plots/blob/main/swiglu/swiglu_speedup_heatmap_bd24_cached_stream_ptr.png?raw=true)|
