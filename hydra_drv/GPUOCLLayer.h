/**
 * \file
 * \brief OpenCL HWLayer
 * 
 *
*/

#include "IHWLayer.h"

#include "../../HydraAPI/clew/clew.h"
#include "../vsgl3/clHelper.h"
#include "../vsgl3/Timer.h"

#include "bitonic_sort_gpu.h"

/** \brief OpenCL HWLayer.
* 
*  This class hold OpenCL buffers (all GPU data) and implement both Path Tracing and MLT on GPU.
*  
*/

class GPUOCLLayer : public CPUSharedData
{
  typedef CPUSharedData Base;

public:

  GPUOCLLayer(int w, int h, int a_flags, int a_deviceId);
  ~GPUOCLLayer();

  void Clear(CLEAR_FLAGS a_flags); ///< Clear internal storage for: MATERIALS, GEOMETRY, LIGHTS, TEXTURES, CUSTOM_DATA

  IMemoryStorage* CreateMemStorage(uint64_t a_maxSizeInBytes, const char* a_name);

  void SetAllBVH4(const ConvertionResult& a_convertedBVH, IBVHBuilder2* a_inBuilderAPI, int a_flags) override;
  void SetAllInstMatrices(const float4x4* a_matrices, int32_t a_matrixNum);
  void SetAllInstLightInstId(const int32_t* a_lightInstIds, int32_t a_instNum);
  void SetAllPODLights(PlainLight* a_lights2, size_t a_number);
  void PrepareEngineGlobals();
  void PrepareEngineTables();
  
  void UpdateVarsOnGPU();


  void GetLDRImage(uint* data, int width, int height) const;
  void GetHDRImage(float4* data, int width, int height) const;

  void ResetPerfCounters();

  void BeginTracingPass();
  void EndTracingPass();
  void FinishAll();

  void InitPathTracing(int seed);
  void ClearAccumulatedColor();

  void ResizeScreen(int w, int h, int a_flags);

  void ContribToExternalImageAccumulator(IHRSharedAccumImage* a_pImage);


  size_t GetMicroThreadsNumber();
  size_t GetAvaliableMemoryAmount(bool allMem);
  size_t GetMemoryTaken();
  MRaysStat GetRaysStat();

  const HRRenderDeviceInfoListElem* ListDevices() const override;

  void SetAllFlagsAndVars(const AllRenderVarialbes& a_vars) override;
  void UpdateConstants();

  void waitIfDebug(const char* file, int line) const;

  void SetNamedBuffer(const char* a_name, void* a_data, size_t a_size);
  void CallNamedFunc(const char* a_name, const char* a_args);

  bool StoreCPUData() const { return m_globals.cpuTrace; }

  bool   FrameBuff_IsAllocated(int a_id) const;  ///< return true if frame buffer is already allocated  (0 - color, 1 - primary, 2 - secondary).
  void   FrameBuff_Alloc(int a_id);              ///< Alloc auxilary frame buffer (0 - color, 1 - primary, 2 - secondary). 
  void   FrameBuff_Free(int a_id);               ///< Free auxilary frame buffer
  void   FrameBuff_SetRenderTarget(int a_id);    ///< Set framebuffe as current render target (0 - color, 1 - primary, 2 - secondary). 
  void   FrameBuff_Blend(int a_buffRes, int a_buff1, int a_buff2, float4 k1, float4 k2); ///< [a_buffRes] = [a_buff1]*k1 + [a_buff2]*k2
  void   FrameBuff_Clear(int a_id);              ///< lear frame buffers
  float4 FrameBuff_AverageColor(int a_id);       ///< calculate average 
  float4 FrameBuff_AverageSqrtColor(int a_id);   ///< Not used. Should be removed.

  bool   MLT_IsAllocated() const;                ///< return true if internal MLT data is allocated
  int    MLT_Alloc(int a_maxBounce);             ///< alloc internal MLT data
  void   MLT_Free();                             ///< free internal MLT DATA

  void   MLT_Init(int a_seed);
  float4 MLT_Burn(int a_iters);
  void   MLT_DoPass();


protected:

  cl_mem FrameBuff_Get(int a_id) const;

  void CreateBuffersGeom(InputGeom a_input, cl_mem_flags a_flags);
  void CreateBuffersBVH(InputGeomBVH a_input, cl_mem_flags a_flags);

  struct SubPixelTempData
  {
    SubPixelTempData() : sppPosBuff(0), pixelsBuff(0), resultBuff(0), resultBuff2(0), resultDS(0), pixelsBuffAppend(0), resultBuffAppend(0), resultBuffAppend2(0), atomicCounter(0) {}
    ~SubPixelTempData()
    {
      if (sppPosBuff) clReleaseMemObject(sppPosBuff); sppPosBuff = 0;
      if (pixelsBuff) clReleaseMemObject(pixelsBuff); pixelsBuff = 0;
      if (resultBuff) clReleaseMemObject(resultBuff); resultBuff = 0;
      if (resultBuff2)clReleaseMemObject(resultBuff2);resultBuff2 = 0;
      if (resultDS)   clReleaseMemObject(resultDS);   resultDS = 0;

      if (pixelsBuffAppend) clReleaseMemObject(pixelsBuffAppend); pixelsBuffAppend = 0;
      if (resultBuffAppend2)clReleaseMemObject(resultBuffAppend2);resultBuffAppend2= 0;
      if (resultBuffAppend) clReleaseMemObject(resultBuffAppend); resultBuffAppend = 0;
      if (atomicCounter)    clReleaseMemObject(atomicCounter);    atomicCounter    = 0;
    }

    cl_mem sppPosBuff;
    cl_mem pixelsBuff;
    cl_mem resultBuff;
    cl_mem resultBuff2;
    cl_mem resultDS;

    cl_mem pixelsBuffAppend;
    cl_mem resultBuffAppend;
    cl_mem resultBuffAppend2;
    cl_mem atomicCounter;

    std::vector<float4> m_clustNormals;
    std::vector<float4> m_clustColors;
    std::vector<int4>   m_clustBegEnd;
  };

  void renderSubPixelData(const char* a_dataName, const std::vector<ushort2>& a_pixels, int spp, float4* a_pixValues, float4* a_subPixValues);
  void renderSubPixelDataBlock(const char* a_dataName, const ushort2* a_pixels, int a_size, int spp, float4* a_pixValues, float4* a_subPixValues, SubPixelTempData& a_tdata);
  void AppendNewPixelClusters(SubPixelTempData& a_data, size_t a_size, int a_spp);

  void memsetu32(cl_mem buff, uint a_val, size_t a_size);
  void memsetf4(cl_mem buff, float4 a_val, size_t a_size, size_t a_offset = 0);
  void memcpyu32(cl_mem buff1, uint a_offset1, cl_mem buff2, uint a_offset2, size_t a_size);

  void float2half(cl_mem buffIn, cl_mem buffOut, size_t a_size);
  void float2half(const std::vector<float>& a_in, std::vector<cl_half>& a_out);
  void float2half(const float* a_inData, size_t a_size, std::vector<cl_half>& a_out);

  void trace1DPrimaryOnly(cl_mem a_rpos, cl_mem a_rdir, cl_mem a_outColor, size_t a_size, size_t a_offset);
  void trace1D(cl_mem a_rpos, cl_mem a_rdir, cl_mem a_outColor, size_t a_size);

  void DrawNormals();
  void AddContributionToScreenGPU(cl_mem in_color, cl_mem in_indices, cl_mem pixWeights, int a_size, int a_width, int a_height, int a_spp,
                                  cl_mem out_colorHDR, cl_mem out_colorLDR);

  void AddContributionToScreenCPU(cl_mem& in_color, cl_mem in_indices, int a_size, int a_width, int a_height, float4* out_color);
  void AddContributionToScreen   (cl_mem& in_color);

  std::vector<uchar4> NormalMapFromDisplacement(int w, int h, const uchar4* a_data, float bumpAmt, bool invHeight, float smoothLvl);
  void Denoise(cl_mem textureIn, cl_mem textureOut, int w, int h, float smoothLvl);

  void inPlaceScanAnySize1f(cl_mem buff, size_t a_size);
  void testScan();
  void testScanFloatsAnySize();
  void testTexture2D();

  void debugSaveFrameBuffer(const char* a_fileName, cl_mem data);
  void debugSaveRays(const char* a_fileName, cl_mem rpos, cl_mem rdir);
  void debugSaveRaysText(const char* a_fileName, cl_mem rpos, cl_mem rdir);
  void debugSaveFloat4Text(const char* a_fileName, cl_mem data);
  
  //
  //
  int    m_initFlags;
  int    m_megaBlockSize;
  int    m_megaBlocksNum;
  int   m_passNumber;
  float m_spp;

  struct CL_SCREEN_BUFFERS
  {
    CL_SCREEN_BUFFERS() : color0(0), pbo(0), m_cpuFrameBuffer(false),
                          targetFrameBuffPointer(0)
    {
      color0CPU.resize(0);
    }

    void free();

    cl_mem color0;  // float4, full screen size
    cl_mem pbo;     // uint,   full screen size

    bool m_cpuFrameBuffer;

    std::vector<float4, aligned16<float4> > color0CPU;


    cl_mem targetFrameBuffPointer;

  } m_screen;

  struct CL_MLT_DATA
  {
    CL_MLT_DATA() : rstateForAcceptReject(0), rstateCurr(0), rstateOld(0), rstateNew(0),
                    xVector(0), yVector(0), xColor(0), yColor(0), zColor(0), oldXY(0), xOldNewId(0), yOldNewId(0),
                    memTaken(0), pssVector(0), samplesLum(0), m_debugSort(false), mppDone(0.0) {}

    cl_mem rstateForAcceptReject; // sizeof(RandGen), MEGABLOCKSIZE size
    cl_mem rstateCurr;            // sizeof(RandGen), MEGABLOCKSIZE size; not allocated, assign m_rays.randGenState
    cl_mem rstateOld;
    cl_mem rstateNew;

    cl_mem xVector;               ///< current vector that store unit hipercube floats
    cl_mem yVector;               ///< next vector that store unit hipercube floats; it should be 0 when MCMC_LAZY is defined; 

    cl_mem xColor;
    cl_mem yColor;
    cl_mem zColor;

    cl_mem oldXY;                 ///< it should be 0 when MCMC_LAZY is defined; 
    cl_mem xOldNewId;
    cl_mem yOldNewId;

    cl_mem pssVector;             // curr pointer to random vector state, used by kernels

    cl_mem samplesLum;
    size_t memTaken;

    Timer  timer;
    double mppDone;
    bool   m_debugSort;

    void free();

  } m_mlt;

  struct CL_BUFFERS_RAYS
  {
    CL_BUFFERS_RAYS() : rayPos(0), rayDir(0), hits(0), rayFlags(0), hitPosNorm(0), hitTexCoord(0), hitMatId(0), hitTangent(0), hitFlatNorm(0), hitPrimSize(0), hitNormUncompressed(0),
                        pathThoroughput(0), pathMisDataPrev(0), pathShadeColor(0), pathAccColor(0), pathAuxColor(0), randGenState(0), pathAuxColorCPU(0),
                        lsam1(0), lsam2(0), shadowRayPos(0), shadowRayDir(0), accPdf(0), oldFlags(0), oldRayDir(0), oldColor(0), lsamProb(0), 
                        lshadow(0), fogAtten(0), samZindex(0), pixWeights(0), MEGABLOCKSIZE(0) {}

    void free();
    void resize(cl_context ctx, cl_command_queue cmdQueue, size_t a_size, bool a_cpuShare, bool a_cpuFB);

    cl_mem rayPos;                   // float4, MEGABLOCKSIZE size
    cl_mem rayDir;                   // float4, MEGABLOCKSIZE size 
    cl_mem hits;
    cl_mem rayFlags;

    cl_mem hitPosNorm;
    cl_mem hitTexCoord;
    cl_mem hitMatId;
    cl_mem hitTangent;
    cl_mem hitFlatNorm;
    cl_mem hitPrimSize;
    cl_mem hitNormUncompressed;

    cl_mem pathThoroughput;
    cl_mem pathMisDataPrev;
    cl_mem pathShadeColor;
    cl_mem pathAccColor;
    cl_mem pathAuxColor;
    cl_mem pathAuxColorCPU;
    cl_mem randGenState;

    cl_mem lsam1;
    cl_mem lsam2;
    cl_mem shadowRayPos;
    cl_mem shadowRayDir;
    cl_mem accPdf;        ///< accumulated pdf weights for 3-way bogolepov light transport

    cl_mem oldFlags;  // store copy of curr bounce data for ConnectEye
    cl_mem oldRayDir; // and cosThetaPrev; store copy of curr bounce data for ConnectEye
    cl_mem oldColor;  // store copy of curr bounce data for ConnectEye

    cl_mem lsamProb;
    cl_mem lshadow;

    cl_mem fogAtten;
    cl_mem samZindex;
    cl_mem pixWeights;

    size_t MEGABLOCKSIZE;

  } m_rays;

  struct CL_GLOBALS
  {
    CL_GLOBALS() : ctx(0), cmdQueue(0), cmdQueueDevToHost(0), platform(0), device(0), m_maxWorkGroupSize(0), oclVer(100), use1DTex(false), liteCore(false),
                   cMortonTable(0), qmcTable(0), devIsCPU(false), cpuTrace(false) {}

    cl_context       ctx;        // OpenCL context
    cl_command_queue cmdQueue;   // OpenCL command que
    cl_command_queue cmdQueueDevToHost;  // OpenCL command que for copying data from GPU to CPU
    cl_platform_id   platform;   // OpenCL platform
    cl_device_id     device;     // OpenCL device

    cl_mem cMortonTable;
    cl_mem qmcTable;              // this is unrelated to previous. Table for Sobo/Niederreiter quasi random sequence.

    size_t m_maxWorkGroupSize;

    int  oclVer;
    bool use1DTex;
    bool liteCore;

    bool devIsCPU;
    bool cpuTrace;

  } m_globals;


  struct CL_SCENE_DATA
  {
    CL_SCENE_DATA() : storageTex(0), storageMat(0), storageGeom(0), storagePdfs(0), storageTexAux(0), matrices(0), instLightInst(0), 
                      matricesSize(0), instLightInstSize(0), allGlobsData(0), allGlobsDataSize(0)
    {
      for (int i = 0; i < MAXBVHTREES; i++)
      {
        bvhBuff    [i] = nullptr;
        objListBuff[i] = nullptr;
        alphTstBuff[i] = nullptr;
        bvhHaveInst[i] = false;
      }
      bvhNumber = 0;
    }

    void free();

    cl_mem storageTex;
    cl_mem storageMat;
    cl_mem storageGeom;
    cl_mem storagePdfs;
    cl_mem storageTexAux;

    cl_mem bvhBuff    [MAXBVHTREES];
    cl_mem objListBuff[MAXBVHTREES];
    cl_mem alphTstBuff[MAXBVHTREES];
    bool   bvhHaveInst[MAXBVHTREES];
    int    bvhNumber;

    cl_mem matrices;
    cl_mem instLightInst;
    size_t matricesSize;
    size_t instLightInstSize;

    cl_mem allGlobsData;
    size_t allGlobsDataSize;

    std::map<std::string, cl_mem> namedBuffers;

  } m_scene;


  struct PROGS
  {
    CLProgram mlt;
    CLProgram screen;
    CLProgram trace;
    CLProgram voxels;
    CLProgram sort;
    CLProgram imagep;
    CLProgram material;
    CLProgram lightp;

  } m_progs;

  enum BIG_MEM_OBJECTS {   // try to account allocated memory, because OpenCL have no such functionality
    MEM_TAKEN_GEOMETRY    = 0,
    MEM_TAKEN_BVH         = 1,
    MEM_TAKEN_TEXTURE1    = 2,
    MEM_TAKEN_TEXTURE2    = 3,
    MEM_TAKEN_TEXTURE3    = 4,
    MEM_TAKEN_TEXTURE4    = 5,
    MEM_TAKEN_RESERVE     = 6,
    MEM_TAKEN_SCREEN      = 7,
    MEM_TAKEN_MLT         = 8,
    MEM_TAKEN_OBJECTS_NUM = 9, // total number of big memory objects 
  };

  size_t m_memoryTaken[MEM_TAKEN_OBJECTS_NUM];


  Timer     m_timer;
  MRaysStat m_stat;

  void runKernel_InitRandomGen(cl_mem a_buffer, size_t a_size, int a_seed);
  void runKernel_MakeEyeRays(cl_mem a_rpos, cl_mem a_rdir, cl_mem a_zindex, cl_mem a_pixWeights, size_t a_size, int a_passNumber);
  void runKernel_MakeLightRays(cl_mem a_rpos, cl_mem a_rdir, cl_mem a_outColor, size_t a_size);
 
  void runKernel_Trace(cl_mem a_rpos, cl_mem a_rdir, cl_mem a_hits, size_t a_size);
  void runKernel_ComputeHit(cl_mem a_rpos, cl_mem a_rdir, size_t a_size);
  void runKernel_NextBounce(cl_mem a_rayFlags, cl_mem a_rpos, cl_mem a_rdir, cl_mem a_outColor, size_t a_size);
  void runKernel_NextTransparentBounce(cl_mem a_rpos, cl_mem a_rdir, cl_mem a_outColor, size_t a_size);

  void ShadePass(cl_mem a_rpos, cl_mem a_rdir, cl_mem a_outColor, size_t a_size, bool a_measureTime);
  void ConnectEyePass(cl_mem in_rayFlags, cl_mem in_hitPos, cl_mem in_hitNorm, cl_mem in_rayDirOld, cl_mem in_color, int a_bounce, size_t a_size);
  void CopyAndPackForConnectEye(cl_mem in_flags, cl_mem in_raydir, cl_mem in_color, cl_mem in_cosPrev,
                                cl_mem out_flags, cl_mem out_raydir, cl_mem out_color, size_t a_size);

  void runKernel_ShadowTrace(cl_mem a_rayFlags, cl_mem a_rpos, cl_mem a_rdir, cl_mem a_outShadow, size_t a_size);
  void runKernel_EyeShadowRays(cl_mem a_rayFlags, cl_mem a_hitPos, cl_mem a_hitNorm, cl_mem a_rpos, cl_mem a_rdir, size_t a_size, int a_haveMaterials);
  void runKernel_ProjectSamplesToScreen(cl_mem a_rayFlags, cl_mem a_hitPos, cl_mem a_hitNorm, cl_mem a_rdir, cl_mem a_rdir2, cl_mem a_colorsIn, cl_mem a_colorsOut, cl_mem a_zindex, size_t a_size, int a_currBounce);

  void runKernel_UpdateForwardPdfFor3Way(cl_mem a_flags, cl_mem old_rayDir, cl_mem next_rayDir, cl_mem acc_pdf, size_t a_size);

  // MLT
  //
  void runKernel_MLTStoreOldXY(cl_mem outXY, cl_mem xVector, size_t a_size);
  void runKernel_MLTEvalContribFunc(cl_mem xVector, cl_mem a_outLum, size_t a_size);
  void runKernel_MLTMakeEyeRaysFromPrimeSpaceSample(int mutateMode, int a_seed, cl_mem rayPos, cl_mem rayDir, cl_mem rstate, cl_mem pssVector, cl_mem color, cl_mem qmcPositions, size_t a_size);

  void runKernel_MLTMakeProposal(cl_mem xVector, cl_mem yVector, cl_mem rstate, bool a_forceLargeStep, size_t a_size);
  void runKernel_MLTMakeIdPairForSorting(cl_mem currGens, cl_mem xOldNewId, cl_mem yOldNewId, cl_mem xVector, cl_mem oldXY, size_t a_size);
  void runKernel_MLTEvalQMCLargeStepIndex(cl_mem rstateBuff, cl_mem positionsBuff, cl_mem counterBuff, size_t a_size);
  void runKernel_MLTInitIdPairForSorting(cl_mem xOldNewId, cl_mem yOldNewId, size_t a_size);
  void runKernel_MLTTestSobolQMC(cl_mem positions, cl_mem outVals, cl_mem qmcTable, size_t a_size);

  void runKernel_MLTContribToScreenAtomics(cl_mem xVector, cl_mem xColor, cl_mem yColor, size_t a_size);
  void runKernel_MLTContribToScreen(cl_mem xOldNewId, cl_mem yOldNewId, cl_mem xColor, cl_mem yColor, size_t a_size);
  void runKernel_MLTAcceptReject(cl_mem rstate, cl_mem xVector, cl_mem yVector, cl_mem xColor, cl_mem yColor, size_t a_size);

  void runKernel_MLTSelectSampleProportionalToContrib(int offset, int offset2, cl_mem rndOld, cl_mem rndOut, cl_mem samplesLum, size_t a_size);
  void runKernel_MLTMoveRandStateByIndex(cl_mem a_to, cl_mem a_from, cl_mem a_to1, cl_mem a_from1, cl_mem a_indices, size_t a_size);
  void runKernel_MLTMoveColorByIndexBack(cl_mem a_to, cl_mem a_from, cl_mem a_indices, size_t a_size);


  // GBuffer and e.t.c
  //
  void runKernel_GenerateSPPRays(cl_mem a_pixels, cl_mem a_sppPos, cl_mem a_rpos, cl_mem a_rdir, size_t a_size, int a_blockSize);
  void runKernel_GetNormalsAndDepth(cl_mem resultBuff, size_t a_size);
  void runKernel_GetTexColor(cl_mem resultBuff, size_t a_size);
  void runKernel_GetGBufferFirstBounce(cl_mem resultBuff, size_t a_size);
  void runKernel_GetAlphaToGBuffer(cl_mem outBuff, cl_mem inBuff, size_t a_size);
  
  void runKernel_ReductionFloat4Average(cl_mem a_src, cl_mem a_dst, size_t a_size, int a_bsize);
  void runKernel_ReductionGBuffer(cl_mem a_src, cl_mem a_dst, size_t a_size, int a_bsize);

  void runKernel_AppendBadPixels(cl_mem a_counter, cl_mem in_data1, cl_mem in_data2, cl_mem out_data1, cl_mem out_data2, cl_mem in_px, cl_mem out_px, size_t a_size, int a_spp);

  float2 runKernel_TestAtomicsPerf(size_t a_size);

  std::vector<ZBlock> m_tempBlocks;
  mutable std::vector<int> m_tempImage;

  bool testSimpleReduction();
  void testDumpRays(const char* a_fNamePos, const char* a_fnameDir);
  void debugDumpF4Buff(const char* a_fNamePos, cl_mem a_buff);

  bool m_clglSharing;

  void runTraceCPU(cl_mem a_rpos, cl_mem a_rdir, cl_mem out_hits, size_t a_size);
  void runTraceShadowCPU(size_t a_size);

  void saveBlocksInfoToFile(cl_mem a_blocks, size_t a_size);

  cl_mem getFrameBuffById(int a_id);
};

void RoundBlocks2D(size_t global_item_size[2], size_t local_item_size[2]);

