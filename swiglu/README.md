# swiGLU PTO-ISA vs torch_npu

## msprof on simulator
|PTO-ISA|torch_npu|
|---------|-----------|
|![PTO-ISA](swiglu_pto-isa.png)|![torch_npu](swiglu_torch.png)|

## msprof on device
|PTO-ISA|torch_npu|
|---------|-----------|
|![PTO-ISA](swiglu_pto-isa_on_device.png)|![torch_npu](swiglu_torch_on_device.png)|

## python benchmark
|Speedup Heatmap|Bandwidth Comparison|
|---------|-----------|
|![Speedup Heatmap](swiglu_speedup_heatmap_bd24.png)|![Bandwidth Comparison](swiglu_tflops_bd24.png)|

PTO-ISA is around `24% faster` than torch_npu for simulated profiling and `13% faster`for on device profiling.

But in the python benchmark it is has `0.78x` the speed of torch_npu.