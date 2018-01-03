#pragma once

#include <vector>

struct SimpleMesh
{
  SimpleMesh(){}

  SimpleMesh(SimpleMesh&& a_in)
  {
    vPos       = std::move(a_in.vPos);
    vNorm      = std::move(a_in.vNorm);
    vTexCoord  = std::move(a_in.vTexCoord);
    triIndices = std::move(a_in.triIndices);
    matIndices = std::move(a_in.matIndices);
  }

  SimpleMesh& operator=(SimpleMesh&& a_in)
  {
    vPos       = std::move(a_in.vPos);
    vNorm      = std::move(a_in.vNorm);
    vTexCoord  = std::move(a_in.vTexCoord);
    triIndices = std::move(a_in.triIndices);
    matIndices = std::move(a_in.matIndices);
    return *this;
  }

  std::vector<float> vPos;
  std::vector<float> vNorm;
  std::vector<float> vTexCoord;
  std::vector<int>   triIndices;
  std::vector<int>   matIndices;
};


SimpleMesh CreatePlane(float a_size = 1.0f);
SimpleMesh CreateCube(float a_size = 1.0f);
SimpleMesh CreateCubeOpen(float a_size = 1.0f);
SimpleMesh CreateSphere(float radius, int numberSlices);
SimpleMesh CreateTorus(float innerRadius, float outerRadius, int numSides, int numFaces);
