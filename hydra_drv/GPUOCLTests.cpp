#include "GPUOCLLayer.h"

#include <iomanip> // for 'std::setfill('0') << std::setw(5)'

void GPUOCLLayer::testScan()
{
  std::cout << ">>=========== scan test =========== " << std::endl;

  std::vector<int> testDataIn(256);
  std::vector<int> testDataOut(256);

  for (int i = 0; i < 256; i++)
  {
    testDataIn[i]  = (i+1)*1;
    testDataOut[i] = 0;
  }

  testDataIn[0] = 1;
  testDataIn[1] = 1;
  testDataIn[2] = 1;
  testDataIn[3] = 5;
  testDataIn[4] = 1;
  testDataIn[5] = 1;


  cl_int ciErr1;

  cl_mem inBuff  = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, testDataIn.size()*sizeof(int), &testDataIn[0], &ciErr1);
  cl_mem outBuff = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, testDataOut.size()*sizeof(int), &testDataOut[0], &ciErr1);

  {
    cl_kernel kernX = m_progs.screen.kernel("ScanTest");
    int isize = 256;
    size_t localWorkSize = 256;
    size_t a_size = 256;

    CHECK_CL(clSetKernelArg(kernX, 0, sizeof(cl_mem), (void*)&inBuff));
    CHECK_CL(clSetKernelArg(kernX, 1, sizeof(cl_mem), (void*)&outBuff));
    CHECK_CL(clSetKernelArg(kernX, 2, sizeof(cl_int), (void*)&isize));

    CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, kernX, 1, NULL, &a_size, &localWorkSize, 0, NULL, NULL));
  }


  CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, outBuff, CL_TRUE, 0, testDataOut.size()*sizeof(int), &testDataOut[0], 0, NULL, NULL));

  std::cout << std::endl;
  for (int i = 0; i < 10; i++)
    std::cout << testDataIn[i] << " ";
  std::cout << std::endl;
  for (int i = 0; i < 10; i++)
    std::cout << testDataOut[i] << " ";
  std::cout << std::endl;

  clReleaseMemObject(inBuff);
  clReleaseMemObject(outBuff);

  std::cout << "<<=========== scan test =========== " << std::endl;
}


void GPUOCLLayer::testScanFloatsAnySize()
{
  std::cout << ">>=========== scan test big =========== " << std::endl;

  std::vector<float> testDataIn(66450);     // 1024*371 + 777, 1024*172 + 777, 512*512/2; // (512*256/2), 66450 is ok
  std::vector<float> testDataOut(testDataIn.size());

  for (int i = 0; i < testDataIn.size(); i++)
  {
    testDataIn[i]  = 1.0f; // rnd(0.0f, 1.0f);
    testDataOut[i] = 0;
  }


  cl_int ciErr1;
  cl_mem outBuff = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, testDataIn.size()*sizeof(float), &testDataIn[0], &ciErr1);

  inPlaceScanAnySize1f(outBuff, testDataIn.size());

  CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, outBuff, CL_TRUE, 0, testDataOut.size()*sizeof(float), &testDataOut[0], 0, NULL, NULL));

  double summ = 0.0f;
  for (int i = 0; i < testDataIn.size(); i++) // perform reference implementation of prefix scan
  {
    summ += double(testDataIn[i]);
    testDataIn[i] = float(summ);
  }

  bool failedOrder = false;
  for (int i = 0; i < testDataOut.size() - 1; i++)
  {
    if (testDataOut[i] - testDataOut[i]*1e-5f > testDataOut[i + 1])
    {
      std::cout << testDataOut[i+0] << std::endl;
      std::cout << testDataOut[i+1] << std::endl;

      failedOrder = true;
      break;
    }
  }

  if (failedOrder)
    std::cout << "test inPlaceScanAnySize1f (order)  have FAILED!" << std::endl;
  else
    std::cout << "test inPlaceScanAnySize1f (order)  have PASSED!" << std::endl;

  bool failed = false;
  for (int i = 0; i < testDataIn.size(); i++)
  {
    float diff = fabs(testDataIn[i] - testDataOut[i]);
    if (diff > fabs(testDataIn[i])*1e-5f)
    {
      std::cout << i << std::endl;
      std::cout << testDataIn[i]  << std::endl;
      std::cout << testDataOut[i] << std::endl;
      failed = true;
      break;
    }
  }

  if (failed)
    std::cout << "test inPlaceScanAnySize1f (values) have FAILED!" << std::endl;
  else
    std::cout << "test inPlaceScanAnySize1f (values) have PASSED!" << std::endl;

  clReleaseMemObject(outBuff);

  std::cout << "<<=========== scan test big =========== " << std::endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////

int align(int a_val, int a_bsize) // may be use this function in future, when aligment requirenments are needed
{
  if (a_val % a_bsize == 0)
    return a_val;
  else
    return (a_val / a_bsize) * a_bsize + a_bsize;
}

void GPUOCLLayer::testTexture2D()
{
  const int N = 264;
  const int M = 259;

  std::vector<int> pixels(N * M);

  for (int y = 0; y < M; y++)
    for (int x = 0; x < N; x++)
      pixels[y*N + x] = 0xFFFFFFFF; // (x + y) % 2 == 0 ? 0 : 0xFFFFFFFF;

  MegaTexData a_data;
  a_data.w = N;
  a_data.h = M;
  a_data.data = (const char*)(&pixels[0]);

  /////////////////////////////////////////////////////////
  cl_int ciErr1 = 0;
  size_t elementByteSize = 4;

  cl_image_format imgFormat;
  imgFormat.image_channel_order = CL_RGBA;
  imgFormat.image_channel_data_type = CL_UNORM_INT8;

  void* inData = (void*)a_data.data;
  /////////////////////////////////////////////////////////

  size_t maxSizeX = 0;
  size_t maxSizeY = 0;
  CHECK_CL(clGetDeviceInfo(m_globals.device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), &maxSizeX, NULL));
  CHECK_CL(clGetDeviceInfo(m_globals.device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), &maxSizeY, NULL));

  if (maxSizeX < a_data.w || maxSizeY < a_data.h)
    RUN_TIME_ERROR("Megatexture exceeds max texture size for OpenCL, developers need to implement resize here");

  cl_mem texture = clCreateImage2D(m_globals.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &imgFormat, size_t(a_data.w), size_t(a_data.h), size_t(a_data.w)*elementByteSize, inData, &ciErr1);

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("Error in clCreateImage2D");

  std::vector<float> resData(N * M);
  cl_mem resBuff = clCreateBuffer(m_globals.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, resData.size()*sizeof(float), &resData[0], &ciErr1);

  if (ciErr1 != CL_SUCCESS)
    RUN_TIME_ERROR("Error in clCreateBuffer");

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  if(true)
  {
    cl_kernel testKern = m_progs.screen.kernel("Texture2DTest");

    size_t global_item_size[2] = { size_t(align(int(N), 16)), size_t(align(int(M), 16)) };
    size_t local_item_size[2]  = {16, 16};

    RoundBlocks2D(global_item_size, local_item_size);

    CHECK_CL(clSetKernelArg(testKern, 0, sizeof(cl_mem), (void*)&texture));
    CHECK_CL(clSetKernelArg(testKern, 1, sizeof(cl_mem), (void*)&resBuff));
    CHECK_CL(clSetKernelArg(testKern, 2, sizeof(cl_int), (void*)&N));
    CHECK_CL(clSetKernelArg(testKern, 3, sizeof(cl_int), (void*)&M));

    CHECK_CL(clEnqueueNDRangeKernel(m_globals.cmdQueue, testKern, 2, NULL, global_item_size, local_item_size, 0, NULL, NULL));
    CHECK_CL(clFinish(m_globals.cmdQueue));
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, resBuff, CL_TRUE, 0, resData.size()*sizeof(int), &resData[0], 0, NULL, NULL));

  std::cout << std::endl;
  std::cout << "resData[center] = " << resData[(M/2)*N + (N/2)] << std::endl;
  std::cout << std::endl;

  clReleaseMemObject(texture);
  clReleaseMemObject(resBuff);
}

void GPUOCLLayer::testDumpRays(const char* a_fNamePos, const char* a_fnameDir)
{
  cvex::vector<float4> temp(m_rays.MEGABLOCKSIZE);
  CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, m_rays.rayPos, CL_TRUE, 0, m_rays.MEGABLOCKSIZE*sizeof(float4), &temp[0], 0, NULL, NULL));

  std::ofstream fout(a_fNamePos, std::ios::binary);
  fout.write((const char*)&temp[0], temp.size()*sizeof(float4));
  fout.close();

  CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, m_rays.rayDir, CL_TRUE, 0, m_rays.MEGABLOCKSIZE*sizeof(float4), &temp[0], 0, NULL, NULL));

  float4 *dits = &temp[0];

  std::ofstream fout2(a_fnameDir, std::ios::binary);
  fout2.write((const char*)&temp[0], temp.size()*sizeof(float4));
  fout2.close();
}

void GPUOCLLayer::debugDumpF4Buff(const char* a_fNamePos, cl_mem a_buff, bool a_textMode)
{
  int size = int(m_rays.MEGABLOCKSIZE);

  cvex::vector<float4> temp(size);
  CHECK_CL(clEnqueueReadBuffer(m_globals.cmdQueue, a_buff, CL_TRUE, 0, size*sizeof(float4), &temp[0], 0, NULL, NULL));

  if(a_textMode)
  {
    std::ofstream fout(a_fNamePos);
    fout << size << std::endl;
    for(int i=0;i<size;i++)
      fout << std::setfill('0') << std::setw(5) << temp[i].x << " " << temp[i].y << " " << temp[i].z << " " << temp[i].w << std::endl;
    fout.close();
  }
  else
  {
    std::ofstream fout(a_fNamePos, std::ios::binary);
    fout.write((const char*)&size, sizeof(int));
    fout.write((const char*)&temp[0], temp.size()*sizeof(float4));
    fout.close();
  }
}

