#define TIMER_GUARDIAN

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
    void            unstart         (void)                  { m_startTicks = -1; }
    float           getElapsed      (void)                  { return ticksToSecs(getElapsedTicks()); }

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

