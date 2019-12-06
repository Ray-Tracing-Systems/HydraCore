//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Baking energy (brightness) BRDF models to create a multi-scattering model, adding them to the single-scattering models.
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <fstream>
#include "../hydra_drv/cmaterial.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////


enum brdfModel { PHONG, TORRSPARR, BECKMANN, GGX, TRGGX };

float inline simpleRnd()
{
  return (rand() % RAND_MAX) / (float)RAND_MAX;
}


float simpleGgxEvalBxDF(const float3 l, const float3 v, const float3 n, const float gloss)
{
  //GGX for light (explicit strategy).
  const float dotNL = dot(n, l);
  const float dotNV = dot(n, v);

  if (dotNV <= 1e-6f || dotNL <= 1e-6f)
    return 0.0f;
    
  const float roughness = 1.0f - gloss;
  const float roughSqr = roughness * roughness;

  const float3 h = normalize(v + l); // half vector.
  const float dotNH = dot(n, h);

  const float D = GGX_Distribution(dotNH, roughSqr);
  const float G = GGX_HeightCorrelatedGeomShadMask(dotNV, dotNL, roughSqr);
  //const float G = GGX_GeomShadMask(dotNV, roughSqr)*GGX_GeomShadMask(dotNL, roughSqr); more correctly than a simple multiplication of masks.
  //const float F = 1.0f;                                                       // Fresnel is not needed here, because it is used for the blend with diffusion.

  return D /** F*/ * G / (4.0f * dotNV * dotNL);  // single-scattering;
}


void BakeBrdfEnergy(const brdfModel brdf, const int maxSample, const int sizeTable, float* result)
{
  // Init
  const float3 n = { 0.0f, 1.0f, 0.0f };


  // Loop for any angle view and any light position

  std::cout << "Calculate BRDF";

  for (int i = 0; i < sizeTable; i++) // angle of view
  {    
    std::cout << ".";

    const float thetaV = (float)i / (float)sizeTable * M_HALFPI;  // angle of view from 0 to PI/2
    const float3 v = { cos(thetaV), sin(thetaV), 0.0f };

    for (int j = 0; j < sizeTable; j++) // gloss
    {      
      const int a = i * sizeTable + j;

      const float gloss = (float)j / (float)sizeTable;

      for (int sample = 0; sample < maxSample; sample++) // random light vector from semisphere
      {         
        const float rnd   = simpleRnd();
        const float phi   = simpleRnd() * M_TWOPI;
        const float h     = sqrt(1.0f - rnd * rnd);

        const float3 l = { cos(phi) * h, rnd, sin(phi) * h };

        result[a] += simpleGgxEvalBxDF(normalize(l), normalize(v), normalize(n), gloss);
      }      

      result[a] /= (float)maxSample;
    }
  }
}







int main(int argc, const char** argv)
{
  const int maxSample   = 100000;
  const int sizeTable   = 16;
  const brdfModel brdf  = GGX;
  
  std::vector<float> result(sizeTable * sizeTable);
    
  BakeBrdfEnergy(GGX, maxSample, sizeTable, result.data()); // Do bake it.
  
  std::cout << std::endl << "Bake complete." << std::endl << std::endl;

  // Print table
  std::cout << std::endl << std::endl << std::endl << std::endl;

  const int widthField = 10;   

  std::cout << "angle: --->";

  for (size_t a = 0; a < sizeTable; a++)
  {
    std::cout.width(widthField);
    std::cout.precision(2);
    std::cout << a / (float)sizeTable * 90.0f;
  }

  std::cout << std::endl;

  for (size_t i = 0; i < sizeTable; i++)
  {
    std::cout << std::endl;
    std::cout << "gloss: " << i / (float)sizeTable;

    for (size_t j = 0; j < sizeTable; j++)
    {
      const int c = i * sizeTable + j;

      std::cout.width(widthField);
      std::cout.precision(2);
      
      std::cout << result[c];
    }
  }

  // Save table to file
  //std::wstring fileName = 

  //std::ofstream fout("cppstudio.txt"); // создаём объект класса ofstream для записи и связываем его с файлом cppstudio.txt
  //fout << "Работа с файлами в С++"; // запись строки в файл
  //fout.close(); // закрываем файл


  std::cout << std::endl << "Save table to file complete." << std::endl;
  std::cout << std::endl;
  
  return 0;
}

