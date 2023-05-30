#include "cl_scan_gpu.h"
#include <vector>

inline size_t sblocksST(size_t elems, int threadsPerBlock)
{
  if (elems % threadsPerBlock == 0 && elems >= threadsPerBlock)
    return elems / threadsPerBlock;
  else
    return (elems / threadsPerBlock) + 1;
}

inline  size_t sRoundBlocks(size_t elems, int threadsPerBlock)
{
  if (elems < threadsPerBlock)
    return (size_t)threadsPerBlock;
  else
    return sblocksST(elems, threadsPerBlock) * threadsPerBlock;
}

struct InternalData
{
  cl_mem tempDataMipLevels[16];
  size_t maxSize;

} g_data;

size_t scan_get_size() { return g_data.maxSize; }

bool scan_alloc_internal(size_t a_size, cl_context ctx)
{
  size_t currSize = a_size;

  for (int i = 0; i < 16; i++)
    g_data.tempDataMipLevels[i] = 0;

  cl_int ciErr1 = CL_SUCCESS;

  for (int i = 0; i < 16; i++)
  {
    int size2 = int(sRoundBlocks(currSize, 256) / 256);

    if (currSize > 0)
    {
      int size3 = size2;
      if (size3 < 256)
        size3 = 256;
      g_data.tempDataMipLevels[i] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size3*sizeof(cl_float) * 4, NULL, &ciErr1);
    }
    else
    {
      g_data.tempDataMipLevels[i] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 256*sizeof(cl_float) * 4, NULL, &ciErr1);
      break;
    }

    currSize = currSize / 256;
  }

  g_data.maxSize = a_size;

  return (ciErr1 == CL_SUCCESS);
}


void scan_free_internal()
{
  for (int i = 0; i < 16; i++)
  {
    if (g_data.tempDataMipLevels[i] != nullptr)
      clReleaseMemObject(g_data.tempDataMipLevels[i]);
    g_data.tempDataMipLevels[i] = 0;
  }
}



void scan1f_gpu(cl_mem a_inBuff, size_t a_size, ScanCLArgs args)
{
  if (g_data.maxSize < a_size)
    return;

  std::vector<size_t> lastSizeV;

  // down, scan phase
  //
  int currMip = 0;
  for (size_t currSize = a_size; currSize > 1; currSize = currSize / 256)
  {
    lastSizeV.push_back(currSize);

    cl_mem inBuff   = (currSize == a_size) ? a_inBuff : g_data.tempDataMipLevels[currMip - 1];
    cl_mem outBuff  = g_data.tempDataMipLevels[currMip];
    cl_kernel kernX = args.scanBlockK;

    int    isize         = int(currSize);
    size_t localWorkSize = 256;
    size_t runSize       = sRoundBlocks(currSize, 256);

   clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&inBuff);
   clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&outBuff);
   clSetKernelArg(kernX, 2, sizeof(cl_int), (void*)&isize);

   clEnqueueNDRangeKernel(args.cmdQueue, kernX, 1, NULL, &runSize, &localWorkSize, 0, NULL, NULL);
   currMip++;
  }

  currMip--;
  //currMip--;
  //lastSizeV.pop_back();

  // up, propagate phase
  //
  while (currMip >= 0)
  {
    size_t currSize = lastSizeV[lastSizeV.size() - 1];
    lastSizeV.pop_back();

    cl_mem inOutBuff = (currMip == 0) ? a_inBuff : g_data.tempDataMipLevels[currMip - 1];
    cl_mem inBuff    = g_data.tempDataMipLevels[currMip];
    cl_kernel kernX  = args.propagateK;

    int    isize         = int(currSize);
    size_t localWorkSize = 256;
    size_t runSize       = sRoundBlocks(currSize, 256);

    clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&inOutBuff);
    clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&inBuff);
    clSetKernelArg(kernX, 2, sizeof(cl_int), (void*)&isize);
    
    clEnqueueNDRangeKernel(args.cmdQueue, kernX, 1, NULL, &runSize, &localWorkSize, 0, NULL, NULL);
    currMip--;
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void reduce_average4f_gpu(cl_mem a_data, size_t a_size, double result[4], ReduceCLArgs args)
{
  cl_kernel kernX = args.reductionK;

  //
  //
  size_t localWorkSize = 256;
  int    isize         = int(a_size);
  a_size               = sRoundBlocks(a_size, 256);

  // a_size => a_size / 256
  //
  clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&a_data);
  clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&g_data.tempDataMipLevels[0]);
  clSetKernelArg(kernX, 2, sizeof(cl_int), (void*)&isize);

  clEnqueueNDRangeKernel(args.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL);

  a_size = a_size / 256;

  if (a_size <= 256)
  {
    // final pass on CPU
    //
    double summ[4] = { 0, 0, 0, 0 };
    std::vector<float> finalColor(a_size*4);

    clEnqueueReadBuffer(args.cmdQueue, g_data.tempDataMipLevels[0], CL_TRUE, 0, finalColor.size()*sizeof(float), &finalColor[0], 0, NULL, NULL);

    for (size_t i = 0; i < a_size; i++)
    {
      summ[0] += double(finalColor[i * 4 + 0]);
      summ[1] += double(finalColor[i * 4 + 1]);
      summ[2] += double(finalColor[i * 4 + 2]);
      summ[3] += double(finalColor[i * 4 + 3]);
    }

    result[0] = summ[0] * (1.0 / double(a_size));
    result[1] = summ[1] * (1.0 / double(a_size));
    result[2] = summ[2] * (1.0 / double(a_size));
    result[3] = summ[3] * (1.0 / double(a_size));
    return;
  }

  // a_size / 256 => a_size / (256*256)
  //
  isize  = int(a_size);
  a_size = sRoundBlocks(a_size, 256);

  clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&g_data.tempDataMipLevels[0]);
  clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&g_data.tempDataMipLevels[1]);
  clSetKernelArg(kernX, 2, sizeof(cl_int), (void*)&isize);
  
  clEnqueueNDRangeKernel(args.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL);


  a_size = a_size / 256;

  // final pass on CPU
  //
  double summ[4] = { 0, 0, 0, 0 };
  std::vector<float> finalColor(a_size * 4);

  cl_int clErr = clEnqueueReadBuffer(args.cmdQueue, g_data.tempDataMipLevels[1], CL_TRUE, 0, finalColor.size()*sizeof(float), &finalColor[0], 0, NULL, NULL);

  for (size_t i = 0; i < a_size; i++)
  {
    summ[0] += double(finalColor[i * 4 + 0]);
    summ[1] += double(finalColor[i * 4 + 1]);
    summ[2] += double(finalColor[i * 4 + 2]);
    summ[3] += double(finalColor[i * 4 + 3]);
  }

  result[0] = summ[0] * (1.0 /double(a_size));
  result[1] = summ[1] * (1.0 /double(a_size));
  result[2] = summ[2] * (1.0 /double(a_size));
  result[3] = summ[3] * (1.0 /double(a_size));
}
