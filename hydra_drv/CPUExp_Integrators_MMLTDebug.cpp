#include <omp.h>
#include "CPUExp_Integrators.h"
#include "time.h"

#include <algorithm> 
#include <cstdio>

bool HR_SaveHDRImageToFileLDR(const wchar_t* a_fileName, int w, int h, const float* a_data, const float a_scaleInv, const float a_gamma = 2.2f);
bool HR_SaveHDRImageToFileHDR(const wchar_t* a_fileName, int w, int h, const float* a_data, const float a_scale = 1.0f);

void IntegratorMMLT::DebugSaveBadPaths()
{
  /////////////////////////////////////////////////////////////////////////////////////////
  //std::cout << "m_debugRaysHeap[0].size() = " << m_debugRaysHeap[0].size() << std::endl;
  //int counter = 0;
  //for (auto p = std::prev(m_debugRaysHeap[0].end()); p != m_debugRaysHeap[0].begin(); --p)
  //{
  //  std::cout << p->first << std::endl;
  //  counter++;
  //  if (counter > 10)
  //    break;
  //}
  /////////////////////////////////////////////////////////////////////////////////////////

  remove("z_rays2.txt");
  remove("z_rands2.txt");

  std::ofstream m_raysFile("z_rays2.txt");
  std::ofstream m_randsFile("z_rands2.txt");

  if (!m_randsFile.is_open())
    std::cout << "piece of shit didn't opened!!!!" << std::endl;

  auto myHeap = m_debugRaysHeap[0];
  for (int i = 1; i < 8; i++)
    myHeap.insert(m_debugRaysHeap[i].begin(), m_debugRaysHeap[i].end());

  const int N = 500;
  m_raysFile  << N << "\n";
  m_randsFile << N << "\n";

  int counter = 0;
  for (auto p = std::prev(myHeap.end()); p != myHeap.begin(); --p)
  {
    auto& vec = p->second.vpos;
    auto& nbs = p->second.randNumbers;

    m_raysFile << vec.size()+1 << "\t";
    for (size_t i = 0; i<vec.size(); i++)
      m_raysFile << vec[i].x << " " << vec[i].y << " " << vec[i].z << " " << vec[i].w << "\n";

    const float4x4 mWorldViewInv = make_float4x4(m_pGlobals->mWorldViewInverse);
    const float3   camPos        = mul(mWorldViewInv, make_float3(0, 0, 0));
    m_raysFile << camPos.x << " " << camPos.y << " " << camPos.z << " " << -1.0f << "\n";

    m_randsFile << nbs.size() << "\t";
    for (size_t i = 0; i < nbs.size(); i++)
      m_randsFile << nbs[i] << " ";
    m_randsFile << "\n";

    if(counter < 10)
      std::cout << "bad contribFunc(sampleColor) = " << p->first << std::endl;

    counter++;
    if (counter > N)
      break;
  }

  m_raysFile.flush();
  m_randsFile.flush();

}

void IntegratorMMLT::DebugLoadPaths()
{
  std::vector< std::vector<float> > randsAll;
  randsAll.reserve(500);

  std::string path = "D:/PROG/HydraAPP/hydra_app/z_rands2.txt";
  std::ifstream fin(path.c_str());
  if (!fin.is_open())
    return;

  int n;
  fin >> n;

  for (int i = 0; i < n; i++)
  {
    int d = 0;
    fin >> d;

    std::vector<float> path(d);
    for (int j = 0; j < d; j++)
    {
      float rayPos;
      fin >> rayPos;
      path[j] = rayPos;
    }

    randsAll.push_back(path);
  }

  int counter = 0;
  for (auto p = randsAll.begin(); p != randsAll.end(); ++p)
  {
    int    xScr = 0, yScr = 0;
    auto   xVec = (*p);

    if (counter == 200)
    {
      int a = 2;
    }

    const int d   = (int(xVec.size()) - MMLT_HEAD_TOTAL_SIZE)/MMLT_FLOATS_PER_BOUNCE;
    float3 yColor = F(xVec, d, (MUTATE_CAMERA | MUTATE_LIGHT), &xScr, &yScr);
    std::cout << contribFunc(yColor) << std::endl;
    counter++;
  }

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


IntegratorMMLT_CompressedRand::PSSampleVC IntegratorMMLT_CompressedRand::Compress(const PSSampleV& a_vec)
{
  PSSampleVC res;

  for(int i=0;i<MMLT_HEAD_TOTAL_SIZE;i++)
    res.head[i] = a_vec[i];

  int sizeRest  = a_vec.size() - MMLT_HEAD_TOTAL_SIZE;
  int bounceNum = sizeRest / MMLT_FLOATS_PER_BOUNCE;

  for(int b=0;b<bounceNum;b++)
  {
    const float* data = a_vec.data() + MMLT_HEAD_TOTAL_SIZE + b*MMLT_FLOATS_PER_BOUNCE;

    float6_gr gr1;
    gr1.group24.x = data[0];
    gr1.group24.y = data[1];
    gr1.group24.z = data[2];
    gr1.group24.w = data[3];
    gr1.group16.x = data[4];
    gr1.group16.y = data[5];

    float4 gr2;
    gr2.x = data[6];
    gr2.y = data[7];
    gr2.z = data[8];
    gr2.w = data[9];

    res.group1[b] = packBounceGroup(gr1);
    res.group2[b] = packBounceGroup2(gr2);
  }

  res.bounceNum = bounceNum;
  return res;
}

IntegratorMMLT_CompressedRand::PSSampleV  IntegratorMMLT_CompressedRand::Decompress(const PSSampleVC& a_vec)
{
  PSSampleV res(randArraySizeOfDepthMMLT(a_vec.bounceNum));
  
  for(int i=0;i<MMLT_HEAD_TOTAL_SIZE;i++)
    res[i] = a_vec.head[i];
  
  for(int b=0;b<a_vec.bounceNum;b++)
  {
    float* data = res.data() + MMLT_HEAD_TOTAL_SIZE + b*MMLT_FLOATS_PER_BOUNCE;

    auto gr1 = unpackBounceGroup (a_vec.group1[b]);
    auto gr2 = unpackBounceGroup2(a_vec.group2[b]);

    data[0] = gr1.group24.x;
    data[1] = gr1.group24.y;
    data[2] = gr1.group24.z;
    data[3] = gr1.group24.w;
    data[4] = gr1.group16.x;
    data[5] = gr1.group16.y;

    data[6] = gr2.x;
    data[7] = gr2.y;
    data[8] = gr2.z;
    data[9] = gr2.w;
  }

  return res;
}

IntegratorMMLT_CompressedRand::PSSampleVC IntegratorMMLT_CompressedRand::InitialSamplePS2(const int d, const int a_burnIters = 0)
{
  auto uncompressed = IntegratorMMLT::InitialSamplePS(d);
  return Compress(uncompressed);
}


void IntegratorMMLT_CompressedRand::DoPassIndirectMLT(int d, float a_bkScale, float4* a_outImage)
{
  auto& gen2 = m_perThread[ThreadId()].gen2;

  //////////////////////////////////////////////////////////////////////////////////// randomize generator
  if (clock() % 3 == 0)
  {
    const int NRandomisation = (clock() % 9) + (clock() % 4);
    for (int i = 0; i < NRandomisation; i++)
      NextState(&gen2);
  }
  //////////////////////////////////////////////////////////////////////////////////// 

  const int samplesPerPass = m_width*m_height;
  mLightSubPathCount = float(samplesPerPass);

  // select seed
  //
  int xScr = 0, yScr = 0;
  auto   xVec   = InitialSamplePS2(d);
  float3 yColor = F(Decompress(xVec), d, (MUTATE_CAMERA | MUTATE_LIGHT), &xScr, &yScr);
  float  y      = contribFunc(yColor);

  // run MCMC
  //
  int accept = 0;
  
  for (int sampleId = 0; sampleId < samplesPerPass; sampleId++)
  {
    int mtype = 0;
    auto xOld  = xVec;
    auto xNew  = Compress( MutatePrimarySpace(Decompress(xOld), d, &mtype) );

    float  yOld      = y;
    float3 yOldColor = yColor;

    int xScrOld = xScr, yScrOld = yScr;
    int xScrNew = 0,    yScrNew = 0;

    float3 yNewColor = F(Decompress(xNew), d, mtype, &xScrNew, &yScrNew);
    float  yNew      = contribFunc(yNewColor);

    float a = (yOld == 0.0f) ? 1.0f : fminf(1.0f, yNew / yOld);

    float p = rndFloat1_Pseudo(&gen2);

    if (p <= a) // accept //
    {
      xVec   = xNew;
      y      = yNew;
      yColor = yNewColor;
      xScr   = xScrNew;
      yScr   = yScrNew;
      accept++;
    }
    else        // reject
    {
      //x      = x;
      //y      = y;
      //yColor = yColor;
    }

    // (5) contrib to image
    //
    float3 contribAtX = a_bkScale*yOldColor*(1.0f / fmaxf(yOld, 1e-6f))*(1.0f - a);
    float3 contribAtY = a_bkScale*yNewColor*(1.0f / fmaxf(yNew, 1e-6f))*a;

    if (dot(contribAtX, contribAtX) > 1e-12f)
    { 
      const int offset = yScrOld*m_width + xScrOld;
      #pragma omp atomic
      a_outImage[offset].x += contribAtX.x;
      #pragma omp atomic
      a_outImage[offset].y += contribAtX.y;
      #pragma omp atomic
      a_outImage[offset].z += contribAtX.z;
      #pragma omp atomic
      a_outImage[offset].w += (1.0f-a);
    }

    if (dot(contribAtY, contribAtY) > 1e-12f)
    { 
      const int offset = yScrNew*m_width + xScrNew;
      #pragma omp atomic
      a_outImage[offset].x += contribAtY.x;
      #pragma omp atomic
      a_outImage[offset].y += contribAtY.y;
      #pragma omp atomic
      a_outImage[offset].z += contribAtY.z;
      #pragma omp atomic
      a_outImage[offset].w += a;
    }
    
  }

  if (omp_get_thread_num() == 0)
  {
    float acceptanceRate = float(accept) / float(samplesPerPass);
    auto oldPrecition = std::cout.precision(3);
    std::cout << "[MMLTC]: acceptanceRate = " << 100.0f*acceptanceRate << "%" << std::endl;
    std::cout.precision(oldPrecition);
  }
}

