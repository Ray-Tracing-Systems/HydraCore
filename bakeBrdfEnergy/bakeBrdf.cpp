//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Baking energy (brightness) BRDF models to create a multi-scattering model, adding them to the single-scattering models.
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <fstream>
#include "../hydra_drv/cmaterial.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////


enum brdfModel { PHONG, TORRSPARR, BECKMANN, GGX, TRGGX, TRANSP};

float SimpleRnd()
{
  return clamp((rand() % RAND_MAX) / (float)(RAND_MAX - 1), 0.0f, 1.0f);
}

template<class T>
void PrintTable(const T * a_result, const int a_widthTable, const int a_heightTable, const int a_depthTable, const bool a_countSample)
{
  if (a_countSample)  
    std::cout << "The number of samples in the table cell." << std::endl << std::endl;
    

  for (size_t z = 0; z < a_depthTable; z++)
  {
    for (size_t y = 0; y < a_heightTable; y++)
    {
      for (size_t x = 0; x < a_widthTable; x++)
      {
        const int size2dTable = a_widthTable * a_heightTable;
        const int zOffset     = z * size2dTable;
        const int a2d         = y * a_widthTable + x;
        const int a3d         = zOffset + a2d;

        std::cout.width(6);

        if (a_countSample && a_result[a3d] > 1000)
          std::cout << a_result[a3d] / 1000 << "K";
        else
        {
          //std::cout.precision(2);
          std::cout << a_result[a3d];
        }
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << std::endl;
  }
}

void CheckForZero(const int* array, const int sizeArray)
{
  for (size_t i = 0; i < sizeArray; i++)
  {
    if (array[i] == 0)    
      std::cout << "Cell " << i << " is empty!" << std::endl;        
  }
}

void GenerateTable(float* msTable, const int sizeArray)
{
  for (size_t i = 0; i < sizeArray; ++i)
    msTable[i] = i;// SimpleRnd();
}

void ReadTable(const float * a_msTable, const float a_roughness, const float a_dotNV, const float a_ior, const int a_widthTable,
               const int a_heightTable, const int a_depthTable)
{
  // Simple access.
  const int size2dTable = a_widthTable * a_heightTable;
  const int size3dTable = size2dTable  * a_depthTable;

  const int x           = (int)(fmin(a_dotNV,     0.9999f) * (float)(a_widthTable));
  const int y           = (int)(fmin(a_roughness, 0.9999f) * (float)(a_heightTable));
  const int cell2d      = y * a_widthTable + x;

  const float iorNormal = (a_ior - 0.4166f) / (2.4f - 0.4166f);        // [0.41, 2.4] -> [0.0, 1.0]
  const int z           = (int)(fmin(iorNormal,   0.9999f) * (float)(a_depthTable));
  const int zOffset     = z * size2dTable;
  const int cell3d      = zOffset + cell2d;

  if (cell3d < size3dTable && a_ior >= 0.4166f && a_ior <= 2.4f)
  {
    //const float Ess = msTable[cell3d];

    // Access with interpolation.
    const float x2      = fmin(a_dotNV,     0.9999f) * (float)(a_widthTable);
    const float y2      = fmin(a_roughness, 0.9999f) * (float)(a_heightTable);
    const float z2      = fmin(iorNormal,   0.9999f) * (float)(a_depthTable);
    const float Ess     = BilinearFrom3dTable(a_msTable, x2, y2, z2, a_widthTable, a_heightTable, a_depthTable, size2dTable);

    //std::cout.precision(2);
    std::cout << "IOR       = " << a_ior       << std::endl;
    std::cout << "iorNormal = " << iorNormal   << std::endl;
    std::cout << "z         = " << z           << std::endl << std::endl;
    std::cout << "roughness = " << a_roughness << std::endl;
    std::cout << "y         = " << y           << std::endl << std::endl;
    std::cout << "dotNV     = " << a_dotNV     << std::endl;
    std::cout << "x         = " << x           << std::endl << std::endl;
    std::cout << "Ess       = " << Ess         << std::endl;
  }
  else
    std::cout << "Error: out of range array! " << "cell3d = " << cell3d << std::endl;
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




void BakeBrdfEnergyTable(float * a_result, const int a_widthTable, const int a_heightTable, const int a_depthTable, const int a_sizeTable, const brdfModel a_brdf, const int a_samplesPerCell)
{
  const double totalSamples = (double)a_samplesPerCell * (double)a_sizeTable;
  const int samplesPerPass  = totalSamples / 100;

  std::cout << "Calculate BRDF " << a_widthTable << " x " << a_heightTable << " x " << a_depthTable << " table." << std::endl;
  std::cout << "Samples per cell: " << a_samplesPerCell / 1000 << "K" << std::endl;
  std::cout << "Total samples:    " << totalSamples / 1000000 << "M" << std::endl;
  std::cout << "Samples per pass: " << samplesPerPass / 1000000.0f << "M" << std::endl;

  const float3 n = { 0.0f, 0.0f, 1.0f };  
  const int size2dTable = a_widthTable * a_heightTable;

  std::vector<int> samplePerCell(a_sizeTable);


  //Loop for any angle view and any light position
  for (int i = 1 ; i <= 100; ++i)
  {
#pragma omp parallel for
    for (int sample = 0; sample < samplesPerPass; ++sample)
    {
      // Reflecion 2D table.
      // Randon roughness.
      const float roughness  = SimpleRnd();
      const float roughSqr   = roughness * roughness;

      // Random ray (eye).
      // We are looking for the angle of theta, for a linear change in the product of vectors (dotNV)
      // and the position of the view vector (v).
      const float cosTheta  = SimpleRnd();
      const float theta     = clamp(M_HALFPI - acos(cosTheta), 0.0f, M_HALFPI);
      const float3 v        = normalize(float3(cos(theta), 0.0f, sin(theta))) * (-1.0f); // v = ray_dir

      // Find cell in 1D array as 2D table.
      const float dotNV     = dot(n, v*(-1.0f));
      const int x           = (int)(fmin(dotNV,     0.9999f) * (float)(a_widthTable));
      const int y           = (int)(fmin(roughness, 0.9999f) * (float)(a_heightTable));
      const int cell2d      = y * a_widthTable + x;

      // Transparency 3D table.
      // Find cell in 1D array as 3D table.
      const float ior       = SimpleRnd() * (2.4f - 0.4166f) + 0.4166f; // [0.41, 2.4]
      const float iorNormal = (ior - 0.4166f) / (2.4f - 0.4166f);       // [0.41, 2.4] -> [0, 1]
      const int z           = (int)(fmin(iorNormal, 0.9999f) * (float)(a_depthTable));
      const int zOffset     = z * size2dTable;
      const int cell3d      = zOffset + cell2d;

      int a = 0;
      float res             = 0;
      const float3 rands(SimpleRnd(), SimpleRnd(), SimpleRnd());


      switch (a_brdf)
      {
        case GGX:    res         = Ggx2017(v, n, roughSqr, rands); a = cell2d; break;
        case TRGGX:  res         = TRGgx(v, n, roughness, rands);  a = cell2d; break;
        case TRANSP:
        {
          const bool  inside     = ior < 1.0f;
          const float outsideIor = inside ? 1.0f / ior : ior;
          res                    = TranspGgx(v, n, roughSqr, inside, outsideIor, rands);
          a                      = cell3d;
          break;
        }
        default: break;
      }


      if (a < a_sizeTable)
      {
        if (!std::isnan(res))
        {
          samplePerCell[a] += 1;
          const int n = samplePerCell[a];
          const float mask = 1.0f / ((float)n + 1.0f);
          a_result[a] = n * a_result[a] * mask + res * mask;
        }
        else
        {
          std::cout << std::endl << "Error: find NaN. Skipping this result.";
          std::cout << "x     = " << x << std::endl;
          std::cout << "y     = " << y << std::endl;
          std::cout << "z     = " << z << std::endl;
          std::cout << "v     = " << v.x << " " << v.y << " " << v.z << std::endl;
          std::cout << "n     = " << n.x << " " << n.y << " " << n.z << std::endl;
          std::cout << "dotNV = " << dotNV << std::endl;
        }
      }
      else
        std::cout << std::endl << "Error: override array size!" << "a = " << a;
    }
    
    std::cout << "Progress: " << i << "% \r";
  }

  // Print number of samples in the table cell.
  std::cout << std::endl << "Bake complete." << std::endl << std::endl;

  //PrintTable(samplePerCell.data(), a_widthTable, a_heightTable, a_depthTable, true);
  CheckForZero(samplePerCell.data(), a_sizeTable);

  std::cout << std::endl;
}


void SaveTableToFile(float * a_array, const int a_widthTable, const int a_heightTable, const int a_depthTable, const int a_areaTable, const brdfModel a_brdf)
{
  std::string a_fileName;

  switch (a_brdf)
  {
    case GGX:           a_fileName = "MSTablesGgx2017.cpp"; break;
    case TRGGX:         a_fileName = "MSTablesTRGgx.cpp";   break;
    case TRANSP:        a_fileName = "MSTablesTransp.cpp";  break;
    default:            a_fileName = "tmp.txt";             break;      
  }

  std::ofstream fout(a_fileName.c_str(), std::ios::binary);

  fout << "// This file is generated automatically in the \"bakeBrdfEnergy\" project." << std::endl << std::endl;
  fout << "// In tables, left to right: dotVN, top to bottom: roughness." << std::endl << std::endl;


  // Print table
  switch (a_brdf)
  {
    case GGX:           fout << "static const float EssGgx2017Table[" << a_areaTable << "] = {" << std::endl; break;
    case TRGGX:         fout << "static const float EssTRGgxTable["   << a_areaTable << "] = {" << std::endl; break;
    case TRANSP:        fout << "static const float EssTranspGgx["    << a_areaTable << "] = {" << std::endl; break;
    default: break;
  }
  
  for (size_t z = 0; z < a_depthTable; z++)
  {
    for (size_t y = 0; y < a_heightTable; y++)
    {
      for (size_t x = 0; x < a_widthTable; x++)
      {
        const int size2dTable = a_widthTable * a_heightTable;
        const int zOffset     = z * size2dTable;
        const int a2d         = y * a_widthTable + x;
        const int a3d         = zOffset + a2d;

        fout.width(8);
        fout.precision(6);
        fout << a_array[a3d];

        if (a3d < (a_areaTable - 1))
          fout << ", ";
      }
      fout << std::endl;
    }
    fout << std::endl;
  }

  fout << "};" << std::endl << std::endl;

  switch (a_brdf)
  {
    case GGX:           fout << "const float* getGgxTable() { return EssGgx2017Table; }";             break;
    case TRGGX:         fout << "const float* getTRGgxTable() { return EssTRGgxTable; }";             break;
    case TRANSP:        fout << "const float* getTranspTable() { return EssTranspGgx; }";             break;
    default: break;
  }

  fout.close(); 

  std::cout << std::endl << "Save table to file complete." << std::endl;
  std::cout << std::endl;
}


////////////////////////////////////////////////////////////////////////////////////////


int main(int argc, const char** argv)
{
  const bool maxQuality  = true;
  const int  widthTable  = 64;
  const int  heightTable = 64;
  int        depthTable  = 1; // for reflect not need 3d table

  const brdfModel brdf   = TRANSP;

  if (brdf == TRANSP) depthTable = 64; // for IOR   
    
  const int sizeTable      = widthTable * heightTable * depthTable;
  const int samplesPerCell = maxQuality ? 100000 : 1000;
  
  std::vector<float> resultTables(sizeTable);
    
  BakeBrdfEnergyTable(resultTables.data(), widthTable, heightTable, depthTable, sizeTable, brdf, samplesPerCell);

  SaveTableToFile(resultTables.data(), widthTable, heightTable, depthTable, sizeTable, brdf);

  //GenerateTable(resultTables.data(), sizeTable);

  //PrintTable(resultTables.data(), widthTable, heightTable, depthTable, false);
    
  //ReadTable(resultTables.data(), 0.2f, 0.7f, 1.5f, widthTable, heightTable, depthTable);

  return 0;
}

