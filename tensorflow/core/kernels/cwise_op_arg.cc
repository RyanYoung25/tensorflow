/* Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {

REGISTER2(UnaryOp, CPU, "Arg", functor::arg, float, double);
#if !defined(__ANDROID__)
REGISTER_KERNEL_BUILDER(Name("ComplexArg").Device(DEVICE_CPU),
                        UnaryOp<CPUDevice, functor::arg<complex64>>);
#endif

#if GOOGLE_CUDA
REGISTER2(UnaryOp, GPU, "Arg", functor::arg, float, double);
#endif
/*
#define REGISTER_KERNEL(T) \ 
REGISTER_KERNEL_BUILDER( \ 
	Name("Arg").Device(DEVICE_CPU).TypeConstraint<T>("T"), \ 
	Arg<CPUDevice, T>); 
REGISTER_KERNEL(complex64);
REGISTER_KERNEL(float); 
REGISTER_KERNEL(double); 
#undef REGISTER_KERNEL

#if GOOGLE_CUDA 
// Forward declarations of the function specializations for GPU (to prevent 
// building the GPU versions here, they will be built compiling _gpu.cu.cc).
// Registration of the GPU implementations. 
#define REGISTER_GPU_KERNEL(T) \ 
REGISTER_KERNEL_BUILDER( \ 
Name("Arg").Device(DEVICE_GPU).TypeConstraint<T>("T"), \ 
Arg<GPUDevice, T>); 
REGISTER_GPU_KERNEL(float); 
REGISTER_GPU_KERNEL(double); 
REGISTER_GPU_KERNEL(complex64); 
#undef REGISTER_GPU_KERNEL 

#endif // GOOGLE_CUDA*/

}  // namespace tensorflow
