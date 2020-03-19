//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Baking energy (brightness) BRDF models to create a multi-scattering model, adding them to the single-scattering models.
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <fstream>
#include <random>
#include "../hydra_drv/cmaterial.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////


enum brdfModel { PHONG, TORRSPARR, BECKMANN, GGX, TRGGX, TRANSP, TRANSP_INSIDE};

float simpleRnd()
{
  return clamp((rand() % RAND_MAX) / (float)(RAND_MAX - 1), 0.0f, 1.0f);
}


float Ggx2017(const float3 ray_dir, const float3 normal, const float roughSqr, const float3 a_rands)
{
  if (roughSqr < 1e-6f)
    return 1.0f;

  const float dotNV = dot(normal, ray_dir * (-1.0f));

  float3 nx, ny, nz   = normal;
  CoordinateSystem(nz, &nx, &ny);

  const float3 wo     = normalize(make_float3(-dot(ray_dir, nx), -dot(ray_dir, ny), -dot(ray_dir, nz)));
  const float3 wh     = GgxVndf(wo, roughSqr, a_rands.x, a_rands.y);
  const float3 wi     = (2.0f * dot(wo, wh) * wh) - wo;       // Compute incident direction by reflecting about wm  
  const float3 newDir = normalize(wi.x * nx + wi.y * ny + wi.z * nz);    // back to normal coordinate system

  const float3 l      = newDir;
  const float dotNL = dot(normal, l);

  if (dotNL < 1e-6f)
    return 0.0f;  
  
  //const float F   = Fresnel is not needed here, because it is used for the blend with diffusion.    
  const float G1    = SmithGGXMasking(dotNV, roughSqr);
  const float G2    = SmithGGXMaskingShadowing(dotNL, dotNV, roughSqr);

  return G2 / fmax(G1, 1e-6f);
}


float TRGgx(const float3 ray_dir, const float3 normal, const float a_roughness, const float3 a_rands)
{
  // To bake anisotropy, you need a 3-dimensional array, but for the sake of simplification,
  // we will bake it as an isotropic one. This will be a good compromise for production.

  if (a_roughness < 1e-6f)
    return 1.0f;

  const float dotNV = dot(normal, ray_dir * (-1.0f));

  if (dotNV < 1e-6f)
    return 0.0f;


  const float glossiness = 1.0f - a_roughness;
  const float rough      = 0.5f - 0.5f * glossiness;
  const float alphax     = BeckmannRoughnessToAlpha(rough*rough);
  const float alphay     = alphax;
  const float2 alpha(alphax, alphay);

  ///////////////////////////////////////////////////////////////////////////// to PBRT coordinate system
  // wo = v = ray_dir
  // wi = l = -newDir
 
  float3 nx, ny, nz = normal;
  CoordinateSystem(nz, &nx, &ny);
  ///////////////////////////////////////////////////////////////////////////// to PBRT coordinate system

  const float3 wo          = normalize(make_float3(-dot(ray_dir, nx), -dot(ray_dir, ny), -dot(ray_dir, nz)));
  const float3 wh          = TrowbridgeReitzDistributionSampleWH(wo, make_float2(a_rands.x, a_rands.y), alpha.x, alpha.y);
  const float3 wi          = normalize((2.0f * dot(wo, wh) * wh) - wo);        // Compute incident direction by reflecting about wh  
  const float3 newDir      = normalize(wi.x * nx + wi.y * ny + wi.z * nz);     // back to normal coordinate system

  const float cosThetaOut  = dot(newDir, normal);
  const float pdf          = TrowbridgeReitzDistributionPdf(wo, wh, alpha.x, alpha.y);

  if (cosThetaOut <= DEPSILON) return 0.0f;
  else                         return TrowbridgeReitzBRDF_PBRT(wo, wi, alpha.x, alpha.y) / fmax(pdf, 1e-6f);
}


float TranspGgx(const float3 a_ray_dir, float3 a_normal, const float a_roughSqr, const bool a_inside, const float a_ior, const float3 a_rands)
{
  const float3  normal2  = a_inside ? (-1.0f) * a_normal : a_normal;
  RefractResult refrData = myRefractGgx(a_ray_dir, normal2, a_ior, 1.0f, a_rands.z);
 
  float Pss              = 1.0f;                          // Pass single-scattering.

  if (a_roughSqr > 0.001f)
  {
    float eta      = 1.0f / a_ior;
    float cosTheta = dot(normal2, a_ray_dir) * (-1.0f);

    if (cosTheta < 0.0f)
      eta  = 1.0f / eta;

    float3 nx, ny, nz    = a_normal;
    CoordinateSystem(nz, &nx, &ny);

    // New sampling Heitz 2017
    const float3 wo      = make_float3(-dot(a_ray_dir, nx), -dot(a_ray_dir, ny), -dot(a_ray_dir, nz));
    const float3 wh      = GgxVndf(wo, a_roughSqr, a_rands.x, a_rands.y);
    const float dotWoWh  = dot(wo, wh);
    float3       newDir  = wo * (-1.0f);

    const float radicand = 1.0f + eta * eta * (dotWoWh * dotWoWh - 1.0f);
    if (radicand > 0.0f)
    {
      newDir             = (eta * dotWoWh - sqrt(radicand)) * wh - eta * wo;    // refract        
      refrData.success   = true;
      refrData.eta       = eta;
    }
    else
    {
      newDir             = 2.0f * dotWoWh * wh - wo;                            // reflect 
      refrData.success   = false;
      refrData.eta       = 1.0f;
    }

    refrData.ray_dir  = normalize(newDir.x * nx + newDir.y * ny + newDir.z * nz);    // back to normal coordinate system

    const float dotNV = fabs(dot(a_normal, a_ray_dir));
    const float dotNL = fabs(dot(a_normal, refrData.ray_dir));

    // Fresnel is not needed here, because it is used for the blend.    
    const float G1    = SmithGGXMasking(dotNV, a_roughSqr);
    const float G2    = SmithGGXMaskingShadowing(dotNL, dotNV, a_roughSqr);
    Pss               = G2 / fmax(G1, 1e-6f);
  }
  
  const float cosThetaOut = dot(refrData.ray_dir, a_normal);

  if      (refrData.success  && cosThetaOut >= -1e-6f) return 0.0f; // refraction/transparency must be under surface!
  else if (!refrData.success && cosThetaOut < 1e-6f)   return 0.0f; // reflection happened in wrong way

  return Pss;
}




void BakeBrdfEnergy(const brdfModel brdf, const int maxSample, const int widthTable, float* result)
{
  std::cout << "Calculate BRDF " << widthTable << "x" << widthTable << " table." << std::endl;
  std::cout << maxSample << " samples" << std::endl;

  const float3 n = { 0.0f, 0.0f, 1.0f };  
  std::vector<int> samplePerCell(widthTable * widthTable);

  //Loop for any angle view and any light position
#pragma omp parallel for
  for (int sample = 0; sample < maxSample; ++sample)
  {
    const float roughness   = clamp(simpleRnd(), 0.0f, 0.9999f);
    const float roughSqr    = roughness * roughness;
    const int i2            = (int)(roughness * (float)(widthTable));

    // We are looking for the angle of theta, for a linear change in the product of vectors (dotNV)
    // and the position of the view vector (v).
    const float cosTheta  = clamp(simpleRnd(), 0.0f, 0.9999f);
    const float theta     = clamp(M_HALFPI - acos(cosTheta), 0.0f, M_HALFPI);
    const float3 v        = normalize(float3(cos(theta), 0.0f, sin(theta))) * (-1.0f); // v = ray_dir

    // Find cell in array.
    const float dotNV     = dot(n, v*(-1.0f));
    const int j2          = (int)(dotNV * (float)(widthTable));
    const int a           = i2 * widthTable + j2;
    
    float res = 0;
    
    const float3 rands(simpleRnd(), simpleRnd(), simpleRnd());
    
    switch (brdf)
    {
      case GGX:    res = Ggx2017(v, n, roughSqr, rands); break;
      case TRGGX:  res = TRGgx(v, n, roughness, rands);   break;
      case TRANSP: 
      {
        const float ior    = 1.5f;
        //const float fresn  = clamp(fresnelReflectionCoeffMentalLike(dotNV, ior), 0.0f, 1.0f);
        const float transp = TranspGgx(v, n, roughSqr, false, ior, rands);
        //const float refl   = Ggx2017(v, n, roughSqr, rands);
        //res                = lerp(transp, refl, fresn);
        res                = transp;
        break;
      }
      case TRANSP_INSIDE:
      {
        const float ior    = 1.5f;
        //const float fresn  = clamp(fresnelReflectionCoeffMentalLike(dotNV, 1.0f), 0.0f, 1.0f);
        const float transp = TranspGgx(v, n, roughSqr, true, ior, rands);
        //const float refl   = Ggx2017(v, n, roughSqr, rands);
        //res                = lerp(transp, refl, fresn);
        res                =  transp;
        break;
      }
      default: break;
    }   

    samplePerCell[a] += 1;
    const int n = samplePerCell[a];
    const float mask = 1.0f / ((float)n + 1.0f);
    result[a] = n * result[a] * mask + res * mask;
  }

  // Print number of samples in the table cell.
  std::cout << std::endl << "Bake complete." << std::endl << std::endl;
  std::cout << "The number of samples in the table cell." << std::endl << std::endl;
  for (size_t i = 0; i < widthTable; i++)
  {
    for (size_t j = 0; j < widthTable; j++)
    {
      const int c = i * widthTable + j;

    if (samplePerCell[c] == 0)
      std::cout << "Cell " << c << " is empty!" << std::endl;

    std::cout.width(6);
      if (samplePerCell[c] >= 1000)
        std::cout << samplePerCell[c] / 1000 << "K";
      else
        std::cout << samplePerCell[c];      
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}


void SaveTableToFile(const brdfModel brdf, const int widthTable, const int areaTable, std::vector<float> result)
{
  std::string a_fileName;

  switch (brdf)
  {
    case GGX:           a_fileName = "MSTablesGgx2017.cpp";      break;
    case TRGGX:         a_fileName = "MSTablesTRGgx.cpp";        break;
    case TRANSP:        a_fileName = "MSTablesTransp.cpp";       break;
    case TRANSP_INSIDE: a_fileName = "MSTablesTranspInside.cpp"; break;
    default:            a_fileName = "tmp.txt";                  break;      
  }

  std::ofstream fout(a_fileName.c_str(), std::ios::binary);

  fout << "// This file is generated automatically in the \"bakeBrdfEnergy\" project." << std::endl << std::endl;
  fout << "// In tables, left to right: dotVN, top to bottom: roughness." << std::endl << std::endl;
  std::cout << "In tables, left to right: dotVN, top to bottom: roughness." << std::endl << std::endl;


  // Print table
  switch (brdf)
  {
    case GGX:           fout << "static const float EssGgx2017Table["    << areaTable << "] = {" << std::endl; break;
    case TRGGX:         fout << "static const float EssTRGgxTable["      << areaTable << "] = {" << std::endl; break;
    case TRANSP:        fout << "static const float EssTranspGgx["       << areaTable << "] = {" << std::endl; break;
    case TRANSP_INSIDE: fout << "static const float EssTranspGgxInside[" << areaTable << "] = {" << std::endl; break;

    default: break;
  }


  for (size_t i = 0; i < widthTable; i++)
  {
    for (size_t j = 0; j < widthTable; j++)
    {
      const int c = i * widthTable + j;

      std::cout.width(6);
      std::cout.precision(2);
      std::cout << result[c];

      fout.width(8);
      fout.precision(6);
      fout << result[c];

      if (c < (areaTable - 1))
        fout << ", ";
    }
    std::cout << std::endl;
    fout << std::endl;
  }

  fout << "};" << std::endl << std::endl;

  switch (brdf)
  {
    case GGX:           fout << "const float* getGgxTable() { return EssGgx2017Table; }";             break;
    case TRGGX:         fout << "const float* getTRGgxTable() { return EssTRGgxTable; }";             break;
    case TRANSP:        fout << "const float* getTranspTable() { return EssTranspGgx; }";             break;
    case TRANSP_INSIDE: fout << "const float* getTranspInsideTable() { return EssTranspGgxInside; }"; break;
    default: break;
  }

  fout.close(); 

  std::cout << std::endl << "Save table to file complete." << std::endl;
  std::cout << std::endl;
}


////////////////////////////////////////////////////////////////////////////////////////


int main(int argc, const char** argv)
{
  const int widthTable = 64;
  const int areaTable  = widthTable * widthTable;
  const int maxSample  = fmin(500000 * areaTable, INT_MAX);
  const brdfModel brdf = TRANSP;
  
  std::vector<float> result(areaTable);

  // Bake table.
  BakeBrdfEnergy(brdf, maxSample, widthTable, result.data()); 

  // Save table to file
  SaveTableToFile(brdf, widthTable, areaTable, result);
  
  return 0;
}

