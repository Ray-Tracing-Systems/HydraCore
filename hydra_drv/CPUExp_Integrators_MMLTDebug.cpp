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
