#include "Timer.h"
#include <iostream>

#ifdef max
#undef max
#endif


#if defined _MSC_VER

//------------------------------------------------------------------------

double Timer::s_ticksToSecsCoef    = -1.0;
long long int Timer::s_prevTicks          = 0;

//------------------------------------------------------------------------

float Timer::end(void)
{
    long long int elapsed = getElapsedTicks();
    m_startTicks += elapsed;
    m_totalTicks += elapsed;
    return ticksToSecs(elapsed);
}

//------------------------------------------------------------------------

inline long long int max(long long int a, long long int b) { return a > b ? a : b; }
inline double max(double a, double b) { return a > b ? a : b; }

long long int Timer::queryTicks(void)
{
    LARGE_INTEGER ticks;
    if (!QueryPerformanceCounter(&ticks))
      throw std::runtime_error("QueryPerformanceFrequency failed");

    s_prevTicks = max(s_prevTicks, ticks.QuadPart);
    return s_prevTicks;
}

//------------------------------------------------------------------------

float Timer::ticksToSecs(long long int ticks)
{
    if (s_ticksToSecsCoef == -1.0)
    {
        LARGE_INTEGER freq;
        if (!QueryPerformanceFrequency(&freq))
          throw std::runtime_error("QueryPerformanceFrequency failed");
        s_ticksToSecsCoef = max(1.0 / (double)freq.QuadPart, 0.0);
    }

    return (float)(ticks * s_ticksToSecsCoef);
}

//------------------------------------------------------------------------

long long int Timer::getElapsedTicks(void)
{
    long long int curr = queryTicks();
    if (m_startTicks == -1)
        m_startTicks = curr;
    return curr - m_startTicks;
}

//------------------------------------------------------------------------

#else

float Timer::getElapsed(void)
{
  double elapsedTime = 0.0;
  
  timeval t2;
  gettimeofday(&t2, NULL);

  timeval t1 = m_timeVal;

  // compute and print the elapsed time in millisec
  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms

  return elapsedTime*0.001f;
}

#endif
