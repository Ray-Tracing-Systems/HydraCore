
#ifndef TIMER_GUARDIAN
#define TIMER_GUARDIAN
#endif

#if defined _MSC_VER

#include <windows.h>
#include <stdexcept>

class Timer
{
public:
    explicit        Timer           (bool started = false)  : m_startTicks(-1), m_totalTicks(0) { if (started) start(); }
                    Timer           (const Timer& other)    { operator=(other); }
                    ~Timer          (void)                  {}

    Timer&          operator=       (const Timer& other)    { m_startTicks = other.m_startTicks; m_totalTicks = other.m_totalTicks; return *this; }

    void            start           (void)                  { m_startTicks = queryTicks(); }
    float           getElapsed      (void)                  { return ticksToSecs(getElapsedTicks()); }

private:

    void            unstart(void)   { m_startTicks = -1; }
    float           end             (void);                 // return elapsed, total += elapsed, restart
    float           getTotal        (void) const            { return ticksToSecs(m_totalTicks); }
    void            clearTotal      (void)                  { m_totalTicks = 0; }

private:

    static long long int queryTicks      (void);
    static float  ticksToSecs     (long long int ticks);
    long long int getElapsedTicks (void);                 // return time since start, start if unstarted

private:
    static double s_ticksToSecsCoef;
    static long long int s_prevTicks;

    long long int   m_startTicks;
    long long int   m_totalTicks;
};

#else

#include <sys/time.h>   // for gettimeofday

class Timer
{
public:
    explicit        Timer           (bool started = false)  { if (started) start(); }
                    Timer           (const Timer& other)    { operator=(other); }
                    ~Timer          (void)                  { }

    Timer&          operator=       (const Timer& other) { m_timeVal = other.m_timeVal; }

    void            start           (void) { gettimeofday(&m_timeVal, 0);}
    float           getElapsed      (void);

private:
    
    timeval m_timeVal;
    
};


#endif
