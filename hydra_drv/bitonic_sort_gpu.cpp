#include "bitonic_sort_gpu.h"


void bitonic_pass_gpu(cl_mem a_buffer, int a_N, int stage, int passOfStage, int a_invertModeOn, BitonicCLArgs other)
{
  const int kernelSize = (a_N >> 1);

  int    iSize  = kernelSize;
  size_t a_size = kernelSize;
  size_t localWorkSize = 256;

  clSetKernelArg(other.bitonicPassK, 0, sizeof(cl_mem), (void*)&a_buffer);
  clSetKernelArg(other.bitonicPassK, 1, sizeof(cl_int), (void*)&stage);
  clSetKernelArg(other.bitonicPassK, 2, sizeof(cl_int), (void*)&passOfStage);
  clSetKernelArg(other.bitonicPassK, 3, sizeof(cl_int), (void*)&a_invertModeOn);

  clEnqueueNDRangeKernel(other.cmdQueue, other.bitonicPassK, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL); 
}


void bitonic_512_gpu(cl_mem a_buffer, int a_N, int stage, int passOfStage, int a_invertModeOn, BitonicCLArgs other)
{
  const int kernelSize = (a_N >> 1);

  int    iSize  = kernelSize;
  size_t a_size = kernelSize;
  size_t localWorkSize = 256;

  clSetKernelArg(other.bitonic512, 0, sizeof(cl_mem), (void*)&a_buffer);
  clSetKernelArg(other.bitonic512, 1, sizeof(cl_int), (void*)&stage);
  clSetKernelArg(other.bitonic512, 2, sizeof(cl_int), (void*)&passOfStage);
  clSetKernelArg(other.bitonic512, 3, sizeof(cl_int), (void*)&a_invertModeOn);
  
  clEnqueueNDRangeKernel(other.cmdQueue, other.bitonic512, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL);
}


void bitonic_sort_gpu_simple(cl_mem a_data, int a_N, BitonicCLArgs other)
{
  int numStages = 0;
  for (int temp = a_N; temp > 2; temp >>= 1)
    numStages++;

  // up, form bitonic sequence with half allays
  //
  for (int stage = 0; stage < numStages; stage++)
  {
    for (int passOfStage = stage; passOfStage >= 0; passOfStage--)
      bitonic_pass_gpu(a_data, a_N, stage, passOfStage, 1, other);
  }

  // down, finally sort it
  //
  for (int passOfStage = numStages; passOfStage >= 0; passOfStage--)
    bitonic_pass_gpu(a_data, a_N, numStages - 1, passOfStage, 0, other);
}

void bitonic_sort_gpu(cl_mem a_data, int a_N, BitonicCLArgs other)
{
  int numStages = 0;
  for (int temp = a_N; temp > 2; temp >>= 1)
    numStages++;

  // up, form bitonic sequence with half allays
  //
  for (int stage = 0; stage < numStages; stage++)
  {
    for (int passOfStage = stage; passOfStage >= 0; passOfStage--)
    {
      if (passOfStage > 0 && passOfStage <= 8)
      {
        bitonic_512_gpu(a_data, a_N, stage, passOfStage, 1, other);
        break;
      }
      else
        bitonic_pass_gpu(a_data, a_N, stage, passOfStage, 1, other);
    }
  }

  // down, finally sort it
  //
  for (int passOfStage = numStages; passOfStage >= 0; passOfStage--)
  {
    if (passOfStage > 0 && passOfStage <= 8)
    {
      bitonic_512_gpu(a_data, a_N, numStages - 1, passOfStage, 0, other);
      break;
    }
    else
      bitonic_pass_gpu(a_data, a_N, numStages - 1, passOfStage, 0, other);
  }

}