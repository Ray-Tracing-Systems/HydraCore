#pragma once

#ifdef WIN32
  #include "../../HydraAPI/utils/clew/clew.h"
#else
  #include <CL/cl.h>
#endif

struct BitonicCLArgs
{
  cl_kernel        bitonic512;
  cl_kernel        bitonic1024;
  cl_kernel        bitonic2048;
  cl_kernel        bitonicPassK;

  cl_command_queue cmdQueue;
  cl_device_id     dev;
};

void bitonic_sort_gpu(cl_mem a_buffer, int a_N, BitonicCLArgs other);
void bitonic_sort_gpu_simple(cl_mem a_data, int a_N, BitonicCLArgs other);