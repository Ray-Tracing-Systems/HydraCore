/**
 \file
 \brief Metropolis Light Transport kernels.

 */

#include "cglobals.h"
#include "crandom.h"
#include "cfetch.h"

#include "clight.h"
#include "cmaterial.h"
#include "cbidir.h"

/**
\brief Evaluate contib function. Not used.
\param in_color   - input color
\param out_colors - output contrib value

*/
__kernel void MLTEvalContribFunc(__global const float4* in_color, __global float* out_colors, int iNumElements)
{
  int tid = GLOBAL_ID_X;
  if (tid >= iNumElements)
    return;

  out_colors[tid] = contribFunc(to_float3(in_color[tid]));
}

__kernel void MMLTCameraPathBounce(__global   float4*        restrict a_rpos,
                                   __global   float4*        restrict a_rdir,
                                   __global   uint*          restrict a_flags,
                                   __global RandomGen*       restrict out_gens,
                         
                                   __global const float4*    restrict in_hitPosNorm,
                                   __global const float2*    restrict in_hitTexCoord,
                                   __global const uint*      restrict in_flatNorm,
                                   __global const HitMatRef* restrict in_matData,
                                   __global const Hit_Part4* restrict in_hitTangent,
                                   __global const float4*    restrict in_hitNormFull,
                                   __global const float4*    restrict in_procTexData,

                                   __global float4*          restrict a_color,
                                   __global float4*          restrict a_normalPrev,       // (!) stote prev normal here, instead of 'a_thoroughput'
                                   __global MisData*         restrict a_misDataPrev,
                                   //__global ushort4*         restrict a_shadow,
                                   __global float4*          restrict a_fog,
                                   //__global const float4*    restrict in_shadeColor,
                                   //__global const float4*    restrict in_emissionColor,
                                   __global PdfVertex*       restrict a_pdfVert,          // (!) MMLT pdfArray
                                   //__global float*           restrict a_camPdfA,    

                                   __global const float4*    restrict in_texStorage1,    
                                   __global const float4*    restrict in_texStorage2,
                                   __global const float4*    restrict in_mtlStorage,
                                   __global const float4*    restrict in_pdfStorage,   //
                                   
                                   int iNumElements,
                                   __global const EngineGlobals*  restrict a_globals)
{

  //__global const PlainLight* pLight = lightAt(a_globals, lightOffset);

}

