#include "Timer.h"
#include <iostream>

#ifdef max
#undef max
#endif

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
