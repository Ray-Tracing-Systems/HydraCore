//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Baking energy (brightness) BRDF models to create a multi-scattering model, adding them to the single-scattering models.
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <fstream>
#include "../hydra_drv/cmaterial.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////


enum brdfModel { PHONG, TORRSPARR, BECKMANN, EXPLICIT_GGX, IMPLICIT_GGX2017, TRGGX };

float simpleRnd()
{
  return clamp((rand() % RAND_MAX) / (float)(RAND_MAX - 1), 0.0f, 1.0f);
}


float ExplicitGGX(const float3 v, const float3 l, const float3 n, const float roughSqr)
{
  const float dotNL = dot(n, l);
  const float dotNV = dot(n, v);

  if (dotNV < 0.0001f || dotNL < 0.0001f || roughSqr < 0.0001f)
    return 1.0f;
  
  const float3 h    = normalize(v + l); // half vector.
  const float dotNH = dot(n, h);

  const float D     = GGX_Distribution(dotNH, roughSqr);
  const float G     = GGX_GeomShadMask(dotNV, roughSqr) * GGX_GeomShadMask(dotNL, roughSqr); // it's more like new sampling Heitz sampling 2017 (GGXSample2AndEvalBRDF).

  return D * G / (4.0f * dotNV * dotNL); 
}

float ImplicitGGX2017(const float3 v, const float3 n, const float roughSqr)
{
  float3 nx, ny, nz   = n;
  CoordinateSystem(nz, &nx, &ny);

  const float r1  = simpleRnd();
  const float r2  = simpleRnd();

  const float3 wo     = normalize(make_float3(-dot(v, nx), -dot(v, ny), -dot(v, nz)));
  const float3 wh     = GgxVndf(wo, roughSqr, r1, r2);
  const float3 wi     = (2.0f * dot(wo, wh) * wh) - wo;       // Compute incident direction by reflecting about wm  
  const float3 newDir = normalize(wi.x * nx + wi.y * ny + wi.z * nz);    // back to normal coordinate system

  const float3 l      = newDir;

  const float dotNL = dot(n, l);
  const float dotNV = dot(n, v * (-1.0f));

  if (dotNV < 1e-6f || dotNL < 1e-6f)
    return 0.0f;  

  if (roughSqr < 1e-6f)
    return 1.0f;
  
  //const float F   = Fresnel is not needed here, because it is used for the blend with diffusion.    
  const float G1    = SmithGGXMasking(dotNV, roughSqr);
  const float G2    = SmithGGXMaskingShadowing(dotNL, dotNV, roughSqr);

  return G2 / G1;
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
    const float3 v        = normalize(float3(cos(theta), 0.0f, sin(theta))) * (-1.0f);

    // Find cell in array.
    const float dotNV     = dot(n, v*(-1.0f));
    const int j2          = (int)(dotNV * (float)(widthTable));
    const int a           = i2 * widthTable + j2;

    //const float r3  = simpleRnd();
    //const float r4  = simpleRnd();
    //const float r5  = simpleRnd();

    //const float phi = M_TWOPI * r3;
    //const float h   = sqrt(1.0f - r4 * r4);
    //const float3 l  = normalize(float3(cos(phi) * h, sin(phi) * h, r2));
    //const float3 l  = normalize(float3(r3, r4, r5));
    
    float res = 0;
    
    switch (brdf)
    {
      //case EXPLICIT_GGX:   res = ExplicitGGX(v, l, n, roughSqr);  break;
      case IMPLICIT_GGX2017: res = ImplicitGGX2017(v, n, roughSqr); break;
      default: break;
    }   

    samplePerCell[a] += 1;
    const int n = samplePerCell[a];
    const float mask = 1.0f / ((float)n + 1.0f);
    result[a] = n * result[a] * mask + res * mask;
  }


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



int main(int argc, const char** argv)
{
  const int widthTable = 64;
  const int areaTable  = widthTable * widthTable;
  const int maxSample  = INT_MAX;
  const brdfModel brdf = IMPLICIT_GGX2017;
  
  std::vector<float> result(areaTable);


  // Bake table.
  BakeBrdfEnergy(brdf, maxSample, widthTable, result.data()); 


  // Save table to file
  std::ofstream fout("MultiScatteringTables.cpp"); 

  fout      << "// This file is generated automatically in the \"bakeBrdfEnergy\" project." << std::endl << std::endl;
  fout      << "// In tables, left to right: dotVN, top to bottom: roughness." << std::endl << std::endl;
  std::cout << "In tables, left to right: dotVN, top to bottom: roughness." << std::endl << std::endl;


  // Print table
  fout << "static const float EssGGXSample2[" << areaTable << "] = {" << std::endl;

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
  fout << "const float* getGGXParams() { return EssGGXSample2; }";
  fout.close(); // закрываем файл

  std::cout << std::endl << "Save table to file complete." << std::endl;
  std::cout << std::endl;
  
  return 0;
}

