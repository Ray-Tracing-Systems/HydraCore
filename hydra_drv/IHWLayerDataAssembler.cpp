#include "IHWLayer.h"

#include <assert.h> 
#include <iostream>
#include <fstream>

#include <algorithm>
#undef min
#undef max

void AllRenderVarialbes::SetVariableI(int a_name, int a_val)
{
  if (a_name<GMAXVARS)
    m_varsI[a_name] = a_val;
}

void AllRenderVarialbes::SetVariableF(int a_name, float a_val)
{
  if (a_name<GMAXVARS)
    m_varsF[a_name] = a_val;
}

void AllRenderVarialbes::SetFlags(unsigned int bits, unsigned int a_value)
{
  if (a_value == 0)
    m_flags = m_flags & (~bits);
  else
    m_flags |= bits;
}

bool AllRenderVarialbes::shadePassEnable(int a_bounce, int a_minBounce, int a_maxBounce)
{
  if ( (m_flags & HRT_DISABLE_SHADING) || (m_flags & HRT_STUPID_PT_MODE))
    return false;

  if (m_varsI[HRT_RENDER_LAYER] == LAYER_POSITIONS || m_varsI[HRT_RENDER_LAYER] == LAYER_NORMALS || m_varsI[HRT_RENDER_LAYER] == LAYER_TEXCOORD || m_varsI[HRT_RENDER_LAYER] == LAYER_TEXCOLOR_AND_MATERIAL)
    return false;

  if (m_varsI[HRT_ENABLE_PATH_REGENERATE] == 0)
  {
    if (m_varsI[HRT_RENDER_LAYER] == LAYER_PRIMARY && a_bounce == 1)
      return false;

    if (m_varsI[HRT_RENDER_LAYER] == LAYER_SECONDARY && a_bounce == 0)
      return false;
  }

  //if((m_flags & HRT_DIRECT_LIGHT_MODE)!=0 && a_bounce > 0)
  //  return false;

  if((m_flags & HRT_INDIRECT_LIGHT_MODE)!=0 && a_bounce == 0)
    return false;

  if(a_bounce == a_maxBounce-1)  // do not evaluate shadow ray for last bounce due to this will actually add one bounce more
    return false;  

  if(a_bounce + 2 < a_minBounce) // do not evaluate bounced that we want to skip
    return false;

  return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

IHWLayer::~IHWLayer()
{
  // don't do this because memory storage objects are deleted by RenderDriverRTE
  //
  // for (auto p = m_allMemStorages.begin(); p != m_allMemStorages.end(); ++p)
  // {
  //   p->second->Clear();
  //   delete p->second;
  //   p->second = nullptr;
  // }

  m_allMemStorages.clear();
}

void IHWLayer::SetAllFlagsAndVars(const AllRenderVarialbes& a_vars)
{
  m_globsBuffHeader.g_flags = a_vars.m_flags;
  m_vars                    = a_vars;
  memcpy(m_globsBuffHeader.varsI, m_vars.m_varsI, sizeof(int)*GMAXVARS);
  memcpy(m_globsBuffHeader.varsF, m_vars.m_varsF, sizeof(float)*GMAXVARS);
}

AllRenderVarialbes IHWLayer::GetAllFlagsAndVars() const
{
  return m_vars;
}


void IHWLayer::SetCamMatrices(float mProjInverse[16], float mWorldViewInverse[16], float mProj[16], float mWorldView[16], float a_aspectX, float a_fovX)
{
  memcpy(m_globsBuffHeader.mProjInverse,      mProjInverse,      sizeof(float) * 16);
  memcpy(m_globsBuffHeader.mWorldViewInverse, mWorldViewInverse, sizeof(float) * 16);
  memcpy(m_globsBuffHeader.mProj,             mProj,             sizeof(float) * 16);
  memcpy(m_globsBuffHeader.mWorldView,        mWorldView,        sizeof(float) * 16);

  float w = float(m_width);
  float h = float(m_height);
 
  AllRenderVarialbes vars = this->GetAllFlagsAndVars();

  vars.m_varsF[HRT_FOV_X] = a_fovX;
  vars.m_varsF[HRT_FOV_Y] = a_fovX/a_aspectX;

  vars.m_varsF[HRT_WIDTH_F]  = w;
  vars.m_varsF[HRT_HEIGHT_F] = h;

  // calc camera params for lt
  //
  const float4x4 mTransform = make_float4x4(mWorldViewInverse);
  const float3   camp1(0, 0, 0);
  const float3   camp2(0, 0, -1);
  const float3   camp1t = mul(mTransform, camp1);
  const float3   camp2t = mul(mTransform, camp2);
  const float3   camFwd = normalize(camp2t - camp1t);
  m_globsBuffHeader.camForward[0] = camFwd.x;
  m_globsBuffHeader.camForward[1] = camFwd.y;
  m_globsBuffHeader.camForward[2] = camFwd.z;

  const float fovX                 = vars.m_varsF[HRT_FOV_X];
  const float tanHalfAngle         = tanf(0.5f*fovX);
  m_globsBuffHeader.imagePlaneDist = w / (2.f * tanHalfAngle);

  this->SetAllFlagsAndVars(vars);

  //std::cerr << "HRT_FOV_X = " << MGML_MATH::RAD_TO_DEG(m_vars.m_varsF[HRT_FOV_X]) << " grads " << std::endl;
  //std::cerr << "HRT_FOV_Y = " << MGML_MATH::RAD_TO_DEG(m_vars.m_varsF[HRT_FOV_Y]) << " grads " << std::endl;
}


size_t CalcConstGlobDataOffsets(EngineGlobals* pGlobals)
{
  const int ALIGN_SIZE = 16;

  size_t currBuffOffset = roundBlocks(sizeof(EngineGlobals) / sizeof(int), ALIGN_SIZE);

  // table for memory storages
  //
  pGlobals->materialsTableOffset   = int(currBuffOffset); currBuffOffset += roundBlocks(pGlobals->materialsTableSize,   ALIGN_SIZE);
  pGlobals->geometryTableOffset    = int(currBuffOffset); currBuffOffset += roundBlocks(pGlobals->geometryTableSize,    ALIGN_SIZE);
  pGlobals->texturesTableOffset    = int(currBuffOffset); currBuffOffset += roundBlocks(pGlobals->texturesTableSize,    ALIGN_SIZE);
  pGlobals->texturesAuxTableOffset = int(currBuffOffset); currBuffOffset += roundBlocks(pGlobals->texturesAuxTableSize, ALIGN_SIZE);
  pGlobals->pdfTableTableOffset    = int(currBuffOffset); currBuffOffset += roundBlocks(pGlobals->pdfTableTableSize,    ALIGN_SIZE);

  pGlobals->lightSelectorTableOffsetRev = int(currBuffOffset); currBuffOffset += roundBlocks(pGlobals->lightSelectorTableSizeRev, ALIGN_SIZE);
  pGlobals->lightSelectorTableOffsetFwd = int(currBuffOffset); currBuffOffset += roundBlocks(pGlobals->lightSelectorTableSizeFwd, ALIGN_SIZE);

  pGlobals->floatArraysOffset = int(currBuffOffset);      currBuffOffset += roundBlocks(pGlobals->floatsArraysSize, ALIGN_SIZE);
  pGlobals->lightsOffset      = int(currBuffOffset);      currBuffOffset += roundBlocks(pGlobals->lightsSize, ALIGN_SIZE);

  return currBuffOffset;
}

void IHWLayer::ResizeTablesForEngineGlobals(int32_t a_geomNum, int32_t a_imgNum, int32_t a_matNum, int32_t a_lightNum)
{
  m_globsBuffHeader.texturesTableSize    = a_imgNum;
  m_globsBuffHeader.texturesAuxTableSize = a_imgNum;
  m_globsBuffHeader.materialsTableSize   = a_matNum;
  m_globsBuffHeader.pdfTableTableSize    = a_lightNum; // + a_matNum*X where X is some estimate of pdf tables per single material ?
  m_globsBuffHeader.geometryTableSize    = a_geomNum;
  m_globsBuffHeader.lightsSize           = int ( (sizeof(PlainLight)*a_lightNum) / sizeof(int) );
}

void IHWLayer::SetAllLightsSelectTable(const float* a_table, int32_t a_tableSize, bool a_fwd)
{
  if (a_fwd)
  {
    m_globsBuffHeader.lightSelectorTableSizeFwd = a_tableSize;
    m_lightSelectTableFwd.resize(a_tableSize);
    if (a_tableSize > 0)
      memcpy(&m_lightSelectTableFwd[0], a_table, a_tableSize * sizeof(float));
  }
  else
  {
    m_globsBuffHeader.lightSelectorTableSizeRev = a_tableSize;
    m_lightSelectTableRev.resize(a_tableSize);
    if (a_tableSize > 0)
      memcpy(&m_lightSelectTableRev[0], a_table, a_tableSize * sizeof(float));
  }
}

void memcpyu32_cpu(int* buff1, uint a_offset1, int* buff2, uint a_offset2, size_t a_size)
{
  int* dst = buff1 + a_offset1;
  int* src = buff2 + a_offset2;
  memcpy(dst, src, a_size*sizeof(int));
}

#define QMC_DOF_FLAG 1
#define QMC_MTL_FLAG 2
#define QMC_LGT_FLAG 4

static void SetQMCVarRemapTable(EngineGlobals *a_globals)
{
  for(int i=0;i<QMC_VARS_NUM;i++)
    a_globals->rmQMC[i] = -1;
  

  int tableVariant = a_globals->varsI[HRT_QMC_VARIANT]; // (0,1,2, ... 7)
  
  if (a_globals->varsI[HRT_ENABLE_DOF] != 1) // if we don't use DOF, exclude it from QMC; reset lowest bit;
  {
    if(tableVariant & QMC_DOF_FLAG != 0)
      tableVariant--;
  }

  // variant 0 (default); screen xy and dof only;
  //
  switch(tableVariant)
  {
    //case 0:
    //  a_globals->rmQMC[QMC_VAR_SCR_X] = 0; // screen xy and dof only;
    //  a_globals->rmQMC[QMC_VAR_SCR_Y] = 1;
    //break;
    
    case 0:                                // well, this is spetial case when we don't want to disable dof; 
    case 1:
      a_globals->rmQMC[QMC_VAR_SCR_X] = 0; // screen xy and dof only;
      a_globals->rmQMC[QMC_VAR_SCR_Y] = 1;
      a_globals->rmQMC[QMC_VAR_DOF_X] = 2;
      a_globals->rmQMC[QMC_VAR_DOF_Y] = 3;
    break;

    case 2:
      a_globals->rmQMC[QMC_VAR_SCR_X] = 0; // screen xy and material;
      a_globals->rmQMC[QMC_VAR_SCR_Y] = 1;
      a_globals->rmQMC[QMC_VAR_MAT_L] = 2;
      a_globals->rmQMC[QMC_VAR_MAT_0] = 3;
      a_globals->rmQMC[QMC_VAR_MAT_1] = 4;
    break;
  
    case 3:                                
      a_globals->rmQMC[QMC_VAR_SCR_X] = 0; // 3 = 2 + 1; screen xy, DOF, MTL
      a_globals->rmQMC[QMC_VAR_SCR_Y] = 1;
      a_globals->rmQMC[QMC_VAR_DOF_X] = 2;
      a_globals->rmQMC[QMC_VAR_DOF_Y] = 3;

      a_globals->rmQMC[QMC_VAR_MAT_L] = 4;
      a_globals->rmQMC[QMC_VAR_MAT_0] = 5;
      a_globals->rmQMC[QMC_VAR_MAT_1] = 6;
    break;

    case 4:                                
      a_globals->rmQMC[QMC_VAR_SCR_X] = 0; // 4; screen xy and light
      a_globals->rmQMC[QMC_VAR_SCR_Y] = 1;
      
      a_globals->rmQMC[QMC_VAR_LGT_N] = 2;
      a_globals->rmQMC[QMC_VAR_LGT_0] = 3;
      a_globals->rmQMC[QMC_VAR_LGT_1] = 4;
      a_globals->rmQMC[QMC_VAR_LGT_2] = 5;
    break;

    case 5:                                
      a_globals->rmQMC[QMC_VAR_SCR_X] = 0; // 4; screen xy, DOF and light
      a_globals->rmQMC[QMC_VAR_SCR_Y] = 1;
      a_globals->rmQMC[QMC_VAR_DOF_X] = 2;
      a_globals->rmQMC[QMC_VAR_DOF_Y] = 3;
      
      a_globals->rmQMC[QMC_VAR_LGT_N] = 4;
      a_globals->rmQMC[QMC_VAR_LGT_0] = 5;
      a_globals->rmQMC[QMC_VAR_LGT_1] = 6;
      a_globals->rmQMC[QMC_VAR_LGT_2] = 7;
    break;
   
    case 6:
      a_globals->rmQMC[QMC_VAR_SCR_X] = 0; // 6 = 4 + 2; screen xy, material and light
      a_globals->rmQMC[QMC_VAR_SCR_Y] = 1;
    
      a_globals->rmQMC[QMC_VAR_MAT_L] = 2;
      a_globals->rmQMC[QMC_VAR_MAT_0] = 3;
      a_globals->rmQMC[QMC_VAR_MAT_1] = 4;
    
      a_globals->rmQMC[QMC_VAR_LGT_N] = 5;
      a_globals->rmQMC[QMC_VAR_LGT_0] = 6;
      a_globals->rmQMC[QMC_VAR_LGT_1] = 7;
      a_globals->rmQMC[QMC_VAR_LGT_2] = 8;
    break;
    
    case 7:
      a_globals->rmQMC[QMC_VAR_SCR_X] = 0; // all of them
      a_globals->rmQMC[QMC_VAR_SCR_Y] = 1;
      a_globals->rmQMC[QMC_VAR_DOF_X] = 2;
      a_globals->rmQMC[QMC_VAR_DOF_Y] = 3;
      
      a_globals->rmQMC[QMC_VAR_MAT_L] = 4;
      a_globals->rmQMC[QMC_VAR_MAT_0] = 5;
      a_globals->rmQMC[QMC_VAR_MAT_1] = 6;
    
      a_globals->rmQMC[QMC_VAR_LGT_N] = 7;
      a_globals->rmQMC[QMC_VAR_LGT_0] = 8;
      a_globals->rmQMC[QMC_VAR_LGT_1] = 9;
      a_globals->rmQMC[QMC_VAR_LGT_2] = 10;
    break;
    
    default:
      a_globals->rmQMC[QMC_VAR_SCR_X] = 0; // screen xy and dof only;
      a_globals->rmQMC[QMC_VAR_SCR_Y] = 1;
      a_globals->rmQMC[QMC_VAR_DOF_X] = 2;
      a_globals->rmQMC[QMC_VAR_DOF_Y] = 3;
    break;
    
  };
  
  
}


void IHWLayer::PrepareEngineGlobals()
{
  size_t totalBuffSize = CalcConstGlobDataOffsets(&m_globsBuffHeader);
  if(m_cdataPrepared.size() < totalBuffSize)
    m_cdataPrepared.resize(totalBuffSize);

  if (m_cdataPrepared.size() == 0)
    return;
  
  SetQMCVarRemapTable(&m_globsBuffHeader);
  
  int* pbuff = &m_cdataPrepared[0];
  memcpy(pbuff, &m_globsBuffHeader, sizeof(EngineGlobals)); 
}

EngineGlobals* IHWLayer::GetEngineGlobals()
{
  return (EngineGlobals*)&m_cdataPrepared[0];
}

void IHWLayer::PrepareEngineTables()
{
  int* pbuff = &m_cdataPrepared[0];

  IMemoryStorage* pTextureStorage  = m_allMemStorages["textures"];
  IMemoryStorage* pTextureStorage2 = m_allMemStorages["textures_aux"];
  IMemoryStorage* pGeomStorage     = m_allMemStorages["geom"];
  IMemoryStorage* pMaterialStorage = m_allMemStorages["materials"];
  IMemoryStorage* pPdfStorage      = m_allMemStorages["pdfs"];

  auto textTable1 = pTextureStorage->GetTable();
  auto textTable2 = pTextureStorage2->GetTable();
  auto geomTable  = pGeomStorage->GetTable();
  auto mateTable  = pMaterialStorage->GetTable();
  auto pdfsTable  = pPdfStorage->GetTable();

  assert( int(textTable1.size()) <= m_globsBuffHeader.texturesTableSize );
  assert( int(textTable2.size()) <= m_globsBuffHeader.texturesAuxTableSize);
  assert( int(geomTable.size())  <= m_globsBuffHeader.geometryTableSize );
  assert( int(mateTable.size())  <= m_globsBuffHeader.materialsTableSize);
  assert( int(pdfsTable.size())  <= m_globsBuffHeader.pdfTableTableSize );

  if(textTable1.size() > 0)
    memcpy(pbuff + m_globsBuffHeader.texturesTableOffset,  &textTable1[0], sizeof(int)*textTable1.size());

  if(textTable2.size() > 0)
    memcpy(pbuff + m_globsBuffHeader.texturesAuxTableOffset, &textTable2[0], sizeof(int)*textTable2.size());

  if (geomTable.size() > 0)
    memcpy(pbuff + m_globsBuffHeader.geometryTableOffset,  &geomTable[0], sizeof(int)*geomTable.size());
  
  if (mateTable.size() > 0)
    memcpy(pbuff + m_globsBuffHeader.materialsTableOffset, &mateTable[0], sizeof(int)*mateTable.size());
  
  if (pdfsTable.size() > 0)
    memcpy(pbuff + m_globsBuffHeader.pdfTableTableOffset,  &pdfsTable[0], sizeof(int)*pdfsTable.size());

  if (m_lightSelectTableRev.size() > 0)
  {
    memcpy(pbuff + m_globsBuffHeader.lightSelectorTableOffsetRev, &m_lightSelectTableRev[0], sizeof(float)*m_lightSelectTableRev.size());
    memcpy(pbuff + m_globsBuffHeader.lightSelectorTableOffsetFwd, &m_lightSelectTableFwd[0], sizeof(float)*m_lightSelectTableFwd.size());
  }
}

void IHWLayer::SetAllPODLights(PlainLight* a_lights2, size_t a_number)
{
  m_globsBuffHeader.lightsSize = int(a_number)*sizeof(PlainLight);
  PrepareEngineGlobals();

  assert(m_cdataPrepared.size() != 0);

  const size_t size1 = sizeof(PlainLight)*a_number;
  const size_t size2 = m_globsBuffHeader.lightsSize;

  assert(size2 >= size1);

  int* pbuff = &m_cdataPrepared[0];
  PlainLight* lightsInBuffer = (PlainLight*)(pbuff + m_globsBuffHeader.lightsOffset);
  memcpy(lightsInBuffer, a_lights2, sizeof(PlainLight)*a_number);

  // #TODO: refactor this code 
  //
  int skyLightOffset = -1; // find sky light 
  {
    for (int i = 0; i < a_number; i++)
    {
      const int* idata = (const int*)(&a_lights2[i].data[0]);
      if (idata[PLIGHT_TYPE] == PLAIN_LIGHT_TYPE_SKY_DOME)
      {
        skyLightOffset = i;
        break;
      }
    }
  }

  // #TODO: find suns and put them to special place in engine globals
  //
  int sunNumber = 0;
  for (int sunId = 0; sunId < MAX_SUN_NUM; sunId++)
  {
    int sunCurrOffset = -1;
    for (int i = 0; i < a_number; i++)
    {
      const int*   idata = (const int*)(&a_lights2[i].data[0]);
      const float* fdata =              &a_lights2[i].data[0];
      if (idata[PLIGHT_TYPE] == PLAIN_LIGHT_TYPE_DIRECT && fdata[DIRECT_LIGHT_SSOFTNESS] > 1e-6f)
      {
        sunCurrOffset = i;
        break;
      }
    }

    if (sunCurrOffset >= 0)
    {
      m_globsBuffHeader.suns[sunId] = a_lights2[sunCurrOffset];
      sunNumber++;
    }
    else
      break;
  }

  m_globsBuffHeader.lightsSize = int(a_number) * sizeof(PlainLight) / sizeof(int);
  m_globsBuffHeader.skyLightId = skyLightOffset;
  m_globsBuffHeader.lightsNum  = int(a_number);
  m_globsBuffHeader.sunNumber  = sunNumber;

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "CPUExp_Integrators.h"


CPUSharedData::CPUSharedData(int w, int h, int a_flags) : m_pIntegrator(nullptr), m_pBVHBuilder(nullptr), m_bvhTreesNum(0)
{
  m_instMatrices  = nullptr;
  m_instMatrixNum = 0;
}

CPUSharedData::~CPUSharedData()
{
  delete m_pIntegrator;
  m_pIntegrator = nullptr;
}


void CPUSharedData::SetAllBVH4(const ConvertionResult& convertedData, IBVHBuilder2* a_inBuilderAPI, int a_flags)
{
  m_pBVHBuilder = a_inBuilderAPI;

  if (m_pBVHBuilder == nullptr)  // if a_inBuilderAPI == nullptr, use converted BVH layout. else use builder itself.
  {
    
    size_t totalmembvh = 0;
    size_t totalmemtri = 0;
    for (int i = 0; i < convertedData.treesNum; i++)
    {
      const float4* ptris = (const float4*)convertedData.pTriangleData[i];

      // copy datas to the temporary storage
      //
      m_bvhTrees[i].m_bvh.assign(convertedData.pBVH[i], convertedData.pBVH[i] + convertedData.nodesNum[i]);
      m_bvhTrees[i].m_tris.assign(ptris, ptris + convertedData.trif4Num[i]);

      if (convertedData.pTriangleAlpha[i] != nullptr)
        m_bvhTrees[i].m_atbl.assign(convertedData.pTriangleAlpha[i], convertedData.pTriangleAlpha[i] + convertedData.triAfNum[i]);

      m_bvhTrees[i].haveInst       = (std::string(convertedData.bvhType[i]) == "object");
      m_bvhTrees[i].smoothOpacity  = (a_flags & BVH_ENABLE_SMOOTH_OPACITY) != 0;

      totalmembvh += convertedData.nodesNum[i] * sizeof(BVHNode);
      totalmemtri += convertedData.trif4Num[i] * sizeof(float4);
      totalmemtri += convertedData.triAfNum[i] * sizeof(int);
    }

    m_bvhTreesNum = convertedData.treesNum;
    //std::cout << "bvh   mem = " << float(totalmembvh) / float(1024 * 1024) << " MB" << std::endl;
    //std::cout << "tri   mem = " << float(totalmemtri) / float(1024 * 1024) << " MB" << std::endl;
    //std::cout << "total mem = " << float(totalmembvh + totalmemtri) / float(1024 * 1024) << " MB" << std::endl;
  }

  IMemoryStorage* pStorage = m_allMemStorages["geom"];

  if (pStorage != nullptr)
    m_geomTable = pStorage->GetTable();
  else
    RUN_TIME_ERROR("CPUSharedData::SetAllBVH4: memory storage for 'geom' not found ");
}

void CPUSharedData::SetAllInstMatrices(const float4x4* a_matrices, int32_t a_matrixNum)
{
  m_instMatrices  = a_matrices;
  m_instMatrixNum = a_matrixNum;
}

void CPUSharedData::SetAllInstLightInstId(const int32_t* a_lightInstIds, int32_t a_instNum)
{
  m_instLightInstId = a_lightInstIds;
  m_instMatrixNum   = a_instNum;
}

void CPUSharedData::SetAllRemapLists(const int* a_allLists, const int2* a_table, int a_allSize, int a_tableSize)
{
  m_remapLists = std::vector<int> (a_allLists, a_allLists + a_allSize);
  m_remapTable = std::vector<int2>(a_table, a_table + a_tableSize);
}

void CPUSharedData::SetAllInstIdToRemapId(const int* a_allInstId, int a_instNum)
{
  m_remapInst  = std::vector<int>(a_allInstId, a_allInstId + a_instNum);
}


void CPUSharedData::PrepareEngineTables()
{
  Base::PrepareEngineTables();
  
  if (m_pIntegrator != nullptr)
  {
    SceneGeomPointers ptrs = CollectPointersForCPUIntegrator();

    IMemoryStorage* pMaterialStorage  = m_allMemStorages["materials"];
    IMemoryStorage* pTexturesStorage1 = m_allMemStorages["textures"];
    IMemoryStorage* pTexturesStorage2 = m_allMemStorages["textures_aux"];
    IMemoryStorage* pPdfsStorage      = m_allMemStorages["pdfs"];

    m_pIntegrator->SetSceneGlobals(m_width, m_height, (EngineGlobals*)&m_cdataPrepared[0]);
    m_pIntegrator->SetSceneGeomPtrs(ptrs);
    m_pIntegrator->SetMaterialStoragePtr((float4*)pMaterialStorage->GetBegin());
    m_pIntegrator->SetTexturesStoragePtr((float4*)pTexturesStorage1->GetBegin());
    m_pIntegrator->SetTexturesStorageAuxPtr((float4*)pTexturesStorage2->GetBegin());
    m_pIntegrator->SetPdfStoragePtr((float4*)pPdfsStorage->GetBegin());

    m_pIntegrator->SetMaterialRemapListPtrs(ptrs.remapListsAll, ptrs.remapListsTab, ptrs.remapInstList,
                                            ptrs.remapAllSize, ptrs.remapTabSize, ptrs.remapInstSize);
  }
}

void CPUSharedData::PrepareEngineGlobals()
{
  Base::PrepareEngineGlobals();

  if (m_cdataPrepared.size() == 0)
    RUN_TIME_ERROR("CPUExpLayer: EngineGlobals were not prepared for some reason ... ?");

  if (m_pIntegrator == nullptr && this->StoreCPUData())
  {
    //m_pIntegrator = new IntegratorStupidPT(m_width, m_height, (EngineGlobals*)&m_cdataPrepared[0]);
    //m_pIntegrator = new IntegratorShadowPT(m_width, m_height, (EngineGlobals*)&m_cdataPrepared[0]);
		//m_pIntegrator = new IntegratorShadowPTSSS(m_width, m_height, (EngineGlobals*)&m_cdataPrepared[0]);
    
    m_pIntegrator = new IntegratorMISPT(m_width, m_height, (EngineGlobals*)&m_cdataPrepared[0], 0);     //#TODO: where m_createFlags gone ???
    //m_pIntegrator = new IntegratorMISPT_QMC(m_width, m_height, (EngineGlobals*)&m_cdataPrepared[0], 0);
    //m_pIntegrator = new IntegratorMISPT_AQMC(m_width, m_height, (EngineGlobals*)&m_cdataPrepared[0], 0);
   
    //m_pIntegrator = new IntegratorLT(m_width, m_height, (EngineGlobals*)&m_cdataPrepared[0]);
    //m_pIntegrator = new IntegratorTwoWay(m_width, m_height, (EngineGlobals*)&m_cdataPrepared[0]);
    //m_pIntegrator = new IntegratorThreeWay(m_width, m_height, (EngineGlobals*)&m_cdataPrepared[0]);

    //m_pIntegrator = new IntegratorSBDPT(m_width, m_height, (EngineGlobals*)&m_cdataPrepared[0]);
    //m_pIntegrator = new IntegratorMMLT(m_width, m_height, (EngineGlobals*)&m_cdataPrepared[0]);
    //m_pIntegrator = new IntegratorMMLT_CompressedRand(m_width, m_height, (EngineGlobals*)&m_cdataPrepared[0]);

    // if (m_vars.m_flags & HRT_ENABLE_MLT)
    //   m_pIntegrator = new IntegratorPSSMLT(m_width, m_height, (EngineGlobals*)&m_cdataPrepared[0], m_createFlags);
    // else if(m_vars.m_flags & HRT_STUPID_PT_MODE)
    //   m_pIntegrator = new IntegratorStupidPT(m_width, m_height, (EngineGlobals*)&m_cdataPrepared[0]);
    // else
    //   m_pIntegrator = new IntegratorMISPT_Full(m_width, m_height, (EngineGlobals*)&m_cdataPrepared[0], m_createFlags);

    std::cout << "[cpu_core]: cpu integrator created" << std::endl;
  }
 
}

SceneGeomPointers CPUSharedData::CollectPointersForCPUIntegrator()
{
  SceneGeomPointers ptrs;

  if (m_pBVHBuilder != nullptr)
  {
    ptrs.pExternalImpl   = m_pBVHBuilder;
    ptrs.matrices        = m_instMatrices;
    ptrs.instLightInstId = m_instLightInstId;
    ptrs.matrixNum       = m_instMatrixNum;
  }
  else
  {
    ptrs.pExternalImpl   = nullptr;
    ptrs.bvhTreesNumber  = m_bvhTreesNum;
    ptrs.matrices        = m_instMatrices;
    ptrs.instLightInstId = m_instLightInstId;
    ptrs.matrixNum       = m_instMatrixNum;

    for (int i = 0; i < ptrs.bvhTreesNumber; i++)
    {
      ptrs.nodesPtr[i] = &m_bvhTrees[i].m_bvh[0];
      ptrs.primsPtr[i] = &m_bvhTrees[i].m_tris[0];

      if (m_bvhTrees[i].m_atbl.size() != 0)
        ptrs.alphaTbl[i] = &m_bvhTrees[i].m_atbl[0];

      ptrs.haveInst[i] = m_bvhTrees[i].haveInst;
    }
  }

  IMemoryStorage* pStorage = m_allMemStorages["geom"];

  if (pStorage != nullptr)
  {
    m_geomTable = pStorage->GetTable();

    if (m_geomTable.size() > 0)
      ptrs.meshes = (const float4*)pStorage->GetBegin();
    else
      RUN_TIME_ERROR("CPUSharedData::CollectPointersForCPUIntegrator: bad memory storage and pointers");
  }
  else
    RUN_TIME_ERROR("CPUSharedData::CollectPointersForCPUIntegrator: bad memory storage and pointers");

  // material remap lists
  //
  if (m_remapInst.size() != 0)
  {
    ptrs.remapInstList = &m_remapInst[0];
    ptrs.remapInstSize = int32_t(m_remapInst.size());
  }
  else
  {
    ptrs.remapInstList = nullptr;
    ptrs.remapInstSize = 0;
  }

  if (m_remapLists.size() != 0)
  {
    ptrs.remapListsAll = &m_remapLists[0];
    ptrs.remapAllSize  = int32_t(m_remapLists.size());
  }
  else
  {
    ptrs.remapListsAll = nullptr;
    ptrs.remapAllSize  = 0;
  }

  if (m_remapTable.size() != 0)
  {
    ptrs.remapListsTab = &m_remapTable[0];
    ptrs.remapTabSize  = int32_t(m_remapTable.size());
  }
  else
  {
    ptrs.remapListsTab = nullptr;
    ptrs.remapTabSize  = 0;
  }

  return ptrs;
}

