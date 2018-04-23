#include "globals_sys.h"


std::string HydraInstallPath()
{
#ifdef WIN32
  return "C:/[Hydra]/bin2/";
#else
  //const std::string installPath2 = "./";
  char user_name[L_cuserid];
  cuserid(user_name);
  std::stringstream ss;
  ss << "/home/" << user_name << "/hydra/";
  return ss.str();
#endif
}

bool isFileExists(const std::string& a_fileName)
{
  std::ifstream fin(a_fileName.c_str());
  return fin.is_open();
}

void PlaneHammersley(float *result, int n)
{
  for (int k = 0; k<n; k++)
  {
    float u = 0;
    int kk = k;

    for (float p = 0.5f; kk; p *= 0.5f, kk >>= 1)
      if (kk & 1)                           // kk mod 2 == 1
        u += p;

    float v = (k + 0.5f) / n;

    result[2 * k + 0] = u;
    result[2 * k + 1] = v;
  }
}


int g_megaTexW[5] = { 0, 0, 0, 0, 0 };
int g_megaTexH[5] = { 0, 0, 0, 0, 0 };

void* g_megaTexData[5] = { nullptr, nullptr, nullptr, nullptr, nullptr };

void cpuGetGlobalTexData(int a_tex, int* pW, int* pH, void** ppData)
{
  (*pW)     = g_megaTexW[a_tex];
  (*pH)     = g_megaTexH[a_tex];
  (*ppData) = g_megaTexData[a_tex];
}

void cpuSetGlobalTexData(int a_tex, int w, int h, void* pData)
{
  g_megaTexW[a_tex]    = w;
  g_megaTexH[a_tex]    = h;
  g_megaTexData[a_tex] = pData;
}


#ifdef WIN32

std::string getWindowsLastErrorMsg()
{
  DWORD dwLastError = GetLastError();

  if (dwLastError != 0)
  {
    char   lpBuffer[256];
    memset(lpBuffer, 0, 256);

    ::FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM,
                     NULL,                                       // No string to be formatted needed
                     dwLastError,                                // Hey Windows: Please explain this error!
                     MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),  // Do it in the standard language
                     lpBuffer,                                   // Put the message here
                     255,                                        // Number of bytes to store the message
                     NULL);

    return std::string(lpBuffer);
  }
  else
    return "";
}

#else

std::string getWindowsLastErrorMsg()
{
  return "";
}


#include <sys/time.h> 

unsigned long GetTickCount()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (tv.tv_sec * 1000 + tv.tv_usec / 1000);
}


#endif


