typedef int2 ElemT;
typedef int  KeyT;
typedef int  ValT;

inline KeyT getKey(ElemT v) { return v.x; }
inline ValT getVal(ElemT v) { return v.y; }

inline bool compare(ElemT a, ElemT b) { return getKey(a) < getKey(b); }

__kernel void bitonic_pass_kernel(__global ElemT* theArray, int stage, int passOfStage, int a_invertModeOn)
{
  int j = get_global_id(0);

  const int r     = 1 << (passOfStage);
  const int lmask = r - 1;

  const int left  = ((j >> passOfStage) << (passOfStage + 1)) + (j & lmask);
  const int right = left + r;

  const ElemT a = theArray[left];
  const ElemT b = theArray[right];

  const bool cmpRes = compare(a, b);

  const ElemT minElem = cmpRes ? a : b;
  const ElemT maxElem = cmpRes ? b : a;

  const int oddEven = j >> stage;

  const bool isSwap = (oddEven & 1) & a_invertModeOn;

  const int minId = isSwap ? right : left;
  const int maxId = isSwap ? left  : right;

  theArray[minId] = minElem;
  theArray[maxId] = maxElem;
}


__kernel void bitonic_512(__global ElemT* theArray, int stage, int passOfStageBegin, int a_invertModeOn)
{
  int tid = get_global_id(0);
  int lid = get_local_id(0);
  
  int blockId = (tid / 256);

  __local ElemT s_array[512];

  s_array[lid + 0]   = theArray[blockId*512 + lid + 0];
  s_array[lid + 256] = theArray[blockId*512 + lid + 256];

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int passOfStage = passOfStageBegin; passOfStage >= 0; passOfStage--)
  {
    const int j     = lid;
    const int r     = 1 << (passOfStage);
    const int lmask = r - 1;

    const int left  = ((j >> passOfStage) << (passOfStage + 1)) + (j & lmask);
    const int right = left + r;

    const ElemT a = s_array[left];
    const ElemT b = s_array[right];

    const bool cmpRes = compare(a, b);

    const ElemT minElem = cmpRes ? a : b;
    const ElemT maxElem = cmpRes ? b : a;

    const int oddEven = tid >> stage; // (j >> stage)

    const bool isSwap = (oddEven & 1) & a_invertModeOn;

    const int minId = isSwap ? right : left;
    const int maxId = isSwap ? left  : right;

    s_array[minId] = minElem;
    s_array[maxId] = maxElem;

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  theArray[blockId*512 + lid + 0]   = s_array[lid + 0];
  theArray[blockId*512 + lid + 256] = s_array[lid + 256];

}

__kernel void bitonic_1024(__global ElemT* theArray, int stage, int passOfStageBegin, int a_invertModeOn)
{
  int tid = get_global_id(0);
  int lid = get_local_id(0);

  int blockId = tid / 512;

  __local ElemT s_array[1024];

  s_array[lid + 0  ] = theArray[blockId * 1024 + lid + 0];
  s_array[lid + 512] = theArray[blockId * 1024 + lid + 512];

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int passOfStage = passOfStageBegin; passOfStage >= 0; passOfStage--)
  {
    const int j = lid;
    const int r = 1 << (passOfStage);
    const int lmask = r - 1;

    const int left = ((j >> passOfStage) << (passOfStage + 1)) + (j & lmask);
    const int right = left + r;

    const ElemT a = s_array[left];
    const ElemT b = s_array[right];

    const bool cmpRes = compare(a, b);

    const ElemT minElem = cmpRes ? a : b;
    const ElemT maxElem = cmpRes ? b : a;

    const int oddEven = tid >> stage; // (j >> stage)

    const bool isSwap = (oddEven & 1) & a_invertModeOn;

    const int minId = isSwap ? right : left;
    const int maxId = isSwap ? left : right;

    s_array[minId] = minElem;
    s_array[maxId] = maxElem;

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  theArray[blockId * 1024 + lid + 0]   = s_array[lid + 0];
  theArray[blockId * 1024 + lid + 512] = s_array[lid + 512];
}


__kernel void bitonic_2048(__global ElemT* theArray, int stage, int passOfStageBegin, int a_invertModeOn)
{
  int tid = get_global_id(0);
  int lid = get_local_id(0);

  int blockId = tid / 1024;

  __local ElemT s_array[2048];

  s_array[lid + 0   ] = theArray[blockId * 2048 + lid + 0];
  s_array[lid + 1024] = theArray[blockId * 2048 + lid + 1024];

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int passOfStage = passOfStageBegin; passOfStage >= 0; passOfStage--)
  {
    const int j = lid;
    const int r = 1 << (passOfStage);
    const int lmask = r - 1;

    const int left = ((j >> passOfStage) << (passOfStage + 1)) + (j & lmask);
    const int right = left + r;

    const ElemT a = s_array[left];
    const ElemT b = s_array[right];

    const bool cmpRes = compare(a, b);

    const ElemT minElem = cmpRes ? a : b;
    const ElemT maxElem = cmpRes ? b : a;

    const int oddEven = tid >> stage; // (j >> stage)

    const bool isSwap = (oddEven & 1) & a_invertModeOn;

    const int minId = isSwap ? right : left;
    const int maxId = isSwap ? left : right;

    s_array[minId] = minElem;
    s_array[maxId] = maxElem;

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  theArray[blockId * 2048 + lid + 0]    = s_array[lid + 0];
  theArray[blockId * 2048 + lid + 1024] = s_array[lid + 1024];
}

