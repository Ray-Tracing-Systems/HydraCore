#include "mesh_utils.h"
#include <cstdint>
#include <math.h>

SimpleMesh CreatePlane(float a_size)
{
  SimpleMesh plane;

  plane.vPos.resize(4 * 4);
  plane.vNorm.resize(4 * 4);
  plane.vTexCoord.resize(4 * 2);
  plane.triIndices.resize(6);
  plane.matIndices.resize(2);

  // plane pos, pos is float4
  //
  plane.vPos[0 * 4 + 0] = -a_size;
  plane.vPos[0 * 4 + 1] = 0.0f;
  plane.vPos[0 * 4 + 2] = -a_size;
  plane.vPos[0 * 4 + 3] = 1.0f; // must be 1

  plane.vPos[1 * 4 + 0] = a_size;
  plane.vPos[1 * 4 + 1] = 0.0f;
  plane.vPos[1 * 4 + 2] = -a_size;
  plane.vPos[1 * 4 + 3] = 1.0f; // must be 1                

  plane.vPos[2 * 4 + 0] = a_size;
  plane.vPos[2 * 4 + 1] = 0.0f;
  plane.vPos[2 * 4 + 2] = a_size;
  plane.vPos[2 * 4 + 3] = 1.0f; // must be 1

  plane.vPos[3 * 4 + 0] = -a_size;
  plane.vPos[3 * 4 + 1] = 0.0f;
  plane.vPos[3 * 4 + 2] = a_size;
  plane.vPos[3 * 4 + 3] = 1.0f; // must be 1

  // plane normals, norm is float4
  //
  plane.vNorm[0 * 4 + 0] = 0.0f;
  plane.vNorm[0 * 4 + 1] = 1.0f;
  plane.vNorm[0 * 4 + 2] = 0.0f;
  plane.vNorm[0 * 4 + 3] = 0.0f; // must be 1

  plane.vNorm[1 * 4 + 0] = 0.0f;
  plane.vNorm[1 * 4 + 1] = 1.0f;
  plane.vNorm[1 * 4 + 2] = 0.0f;
  plane.vNorm[1 * 4 + 3] = 0.0f; // must be 1

  plane.vNorm[2 * 4 + 0] = 0.0f;
  plane.vNorm[2 * 4 + 1] = 1.0f;
  plane.vNorm[2 * 4 + 2] = 0.0f;
  plane.vNorm[2 * 4 + 3] = 0.0f; // must be 1

  plane.vNorm[3 * 4 + 0] = 0.0f;
  plane.vNorm[3 * 4 + 1] = 1.0f;
  plane.vNorm[3 * 4 + 2] = 0.0f;
  plane.vNorm[3 * 4 + 3] = 0.0f; // must be 1

  // plane texture coords
  //
  plane.vTexCoord[0 * 2 + 0] = 0.0f;
  plane.vTexCoord[0 * 2 + 1] = 0.0f;

  plane.vTexCoord[1 * 2 + 0] = 1.0f;
  plane.vTexCoord[1 * 2 + 1] = 0.0f;

  plane.vTexCoord[2 * 2 + 0] = 1.0f;
  plane.vTexCoord[2 * 2 + 1] = 1.0f;

  plane.vTexCoord[3 * 2 + 0] = 0.0f;
  plane.vTexCoord[3 * 2 + 1] = 1.0f;

  // construct triangles
  //
  plane.triIndices[0] = 0;
  plane.triIndices[1] = 1;
  plane.triIndices[2] = 2;

  plane.triIndices[3] = 0;
  plane.triIndices[4] = 2;
  plane.triIndices[5] = 3;

  // assign material per triangle
  //
  plane.matIndices[0] = 0;
  plane.matIndices[1] = 0;

  return plane;
}


SimpleMesh CreateCube(float a_size)
{
  uint32_t numberVertices = 24;
  uint32_t numberIndices  = 36;

  float cubeVertices[] =
  {
    -1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, +1.0f,
    +1.0f, -1.0f, +1.0f, +1.0f,
    +1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, +1.0f,
    +1.0f, +1.0f, +1.0f, +1.0f,
    +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, +1.0f, -1.0f, +1.0f,
    +1.0f, +1.0f, -1.0f, +1.0f,
    +1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, +1.0f,
    +1.0f, +1.0f, +1.0f, +1.0f,
    +1.0f, -1.0f, +1.0f, +1.0f,
    -1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, +1.0f,
    -1.0f, +1.0f, -1.0f, +1.0f,
    +1.0f, -1.0f, -1.0f, +1.0f,
    +1.0f, -1.0f, +1.0f, +1.0f,
    +1.0f, +1.0f, +1.0f, +1.0f,
    +1.0f, +1.0f, -1.0f, +1.0f
  };

  float cubeNormals[] =
  {
    0.0f, -1.0f, 0.0f, +1.0f,
    0.0f, -1.0f, 0.0f, +1.0f,
    0.0f, -1.0f, 0.0f, +1.0f,
    0.0f, -1.0f, 0.0f, +1.0f,
    0.0f, +1.0f, 0.0f, +1.0f,
    0.0f, +1.0f, 0.0f, +1.0f,
    0.0f, +1.0f, 0.0f, +1.0f,
    0.0f, +1.0f, 0.0f, +1.0f,
    0.0f, 0.0f, -1.0f, +1.0f,
    0.0f, 0.0f, -1.0f, +1.0f,
    0.0f, 0.0f, -1.0f, +1.0f,
    0.0f, 0.0f, -1.0f, +1.0f,
    0.0f, 0.0f, +1.0f, +1.0f,
    0.0f, 0.0f, +1.0f, +1.0f,
    0.0f, 0.0f, +1.0f, +1.0f,
    0.0f, 0.0f, +1.0f, +1.0f,
    -1.0f, 0.0f, 0.0f, +1.0f,
    -1.0f, 0.0f, 0.0f, +1.0f,
    -1.0f, 0.0f, 0.0f, +1.0f,
    -1.0f, 0.0f, 0.0f, +1.0f,
    +1.0f, 0.0f, 0.0f, +1.0f,
    +1.0f, 0.0f, 0.0f, +1.0f,
    +1.0f, 0.0f, 0.0f, +1.0f,
    +1.0f, 0.0f, 0.0f, +1.0f
  };

  float cubeTexCoords[] =
  {
    0.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 1.0f,
    1.0f, 0.0f,
    1.0f, 0.0f,
    1.0f, 1.0f,
    0.0f, 1.0f,
    0.0f, 0.0f,
    0.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 1.0f,
    1.0f, 0.0f,
    0.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 1.0f,
    1.0f, 0.0f,
    0.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 1.0f,
    1.0f, 0.0f,
    0.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 1.0f,
    1.0f, 0.0f,
  };

  uint32_t cubeIndices[] =
  {
    0, 2, 1,
    0, 3, 2,
    4, 5, 6,
    4, 6, 7,
    8, 9, 10,
    8, 10, 11,
    12, 15, 14,
    12, 14, 13,
    16, 17, 18,
    16, 18, 19,
    20, 23, 22,
    20, 22, 21
  };

  SimpleMesh cube;

  cube.vPos.resize(numberVertices*4);
  cube.vNorm.resize(numberVertices*4);
  cube.vTexCoord.resize(numberVertices*2);
  cube.triIndices.resize(numberIndices);
  cube.matIndices.resize(numberIndices/3);

  for (size_t i = 0; i < cube.vPos.size(); i++)
    cube.vPos[i] = cubeVertices[i] * a_size;

  for (size_t i = 0; i < cube.vNorm.size(); i++)
    cube.vNorm[i] = cubeNormals[i];

  for (size_t i = 0; i < cube.vTexCoord.size(); i++)
    cube.vTexCoord[i] = cubeTexCoords[i];

  for (size_t i = 0; i < cube.triIndices.size(); i++)
    cube.triIndices[i] = cubeIndices[i];

  for (size_t i = 0; i < cube.matIndices.size(); i++)
    cube.matIndices[i] = 0;

  return cube;
}

SimpleMesh CreateCubeOpen(float a_size)
{
  int numberVertices = 24;
  int numberIndices  = 36 - 6;

  float cubeVertices[] =
  {
    -1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, +1.0f,
    +1.0f, -1.0f, +1.0f, +1.0f,
    +1.0f, -1.0f, -1.0f, +1.0f,

    -1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, +1.0f,
    +1.0f, +1.0f, +1.0f, +1.0f,
    +1.0f, +1.0f, -1.0f, +1.0f,

    -1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, +1.0f, -1.0f, +1.0f,
    +1.0f, +1.0f, -1.0f, +1.0f,
    +1.0f, -1.0f, -1.0f, +1.0f,

    -1.0f, -1.0f, +1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, +1.0f,
    +1.0f, +1.0f, +1.0f, +1.0f,
    +1.0f, -1.0f, +1.0f, +1.0f,

    -1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, +1.0f,
    -1.0f, +1.0f, -1.0f, +1.0f,
    +1.0f, -1.0f, -1.0f, +1.0f,

    +1.0f, -1.0f, +1.0f, +1.0f,
    +1.0f, +1.0f, +1.0f, +1.0f,
    +1.0f, +1.0f, -1.0f, +1.0f
  };


  float cubeNormals[] =
  {
    0.0f, -1.0f, 0.0f, 0.0f, 
    0.0f, -1.0f, 0.0f, 0.0f,
    0.0f, -1.0f, 0.0f, 0.0f,
    0.0f, -1.0f, 0.0f, 0.0f,
                       
    0.0f, +1.0f, 0.0f, 0.0f,
    0.0f, +1.0f, 0.0f, 0.0f,
    0.0f, +1.0f, 0.0f, 0.0f,
    0.0f, +1.0f, 0.0f, 0.0f,
   
    0.0f, 0.0f, -1.0f, 0.0f,
    0.0f, 0.0f, -1.0f, 0.0f,
    0.0f, 0.0f, -1.0f, 0.0f,
    0.0f, 0.0f, -1.0f, 0.0f,
           
    0.0f, 0.0f, +1.0f, 0.0f,
    0.0f, 0.0f, +1.0f, 0.0f,
    0.0f, 0.0f, +1.0f, 0.0f,
    0.0f, 0.0f, +1.0f, 0.0f,
            
    -1.0f, 0.0f, 0.0f, 0.0f,
    -1.0f, 0.0f, 0.0f, 0.0f,
    -1.0f, 0.0f, 0.0f, 0.0f,
    -1.0f, 0.0f, 0.0f, 0.0f,
                   
    +1.0f, 0.0f, 0.0f, 0.0f,
    +1.0f, 0.0f, 0.0f, 0.0f,
    +1.0f, 0.0f, 0.0f, 0.0f,
    +1.0f, 0.0f, 0.0f, 0.0f
  };

  float cubeTangents[] =
  {
    -1.0f, 0.0f, 0.0f,
    -1.0f, 0.0f, 0.0f,
    -1.0f, 0.0f, 0.0f,
    -1.0f, 0.0f, 0.0f,

    +1.0f, 0.0f, 0.0f,
    +1.0f, 0.0f, 0.0f,
    +1.0f, 0.0f, 0.0f,
    +1.0f, 0.0f, 0.0f,

    -1.0f, 0.0f, 0.0f,
    -1.0f, 0.0f, 0.0f,
    -1.0f, 0.0f, 0.0f,
    -1.0f, 0.0f, 0.0f,

    +1.0f, 0.0f, 0.0f,
    +1.0f, 0.0f, 0.0f,
    +1.0f, 0.0f, 0.0f,
    +1.0f, 0.0f, 0.0f,

    0.0f, 0.0f, +1.0f,
    0.0f, 0.0f, +1.0f,
    0.0f, 0.0f, +1.0f,
    0.0f, 0.0f, +1.0f,

    0.0f, 0.0f, -1.0f,
    0.0f, 0.0f, -1.0f,
    0.0f, 0.0f, -1.0f,
    0.0f, 0.0f, -1.0f
  };

  float cubeTexCoords[] =
  {
    0.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 1.0f,
    1.0f, 0.0f,

    1.0f, 0.0f,
    1.0f, 1.0f,
    0.0f, 1.0f,
    0.0f, 0.0f,

    0.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 1.0f,
    1.0f, 0.0f,

    0.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 1.0f,
    1.0f, 0.0f,

    0.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 1.0f,
    1.0f, 0.0f,

    0.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 1.0f,
    1.0f, 0.0f,
  };

  int cubeIndices[] =
  {
    0, 2, 1,
    0, 3, 2,

    4, 5, 6,
    4, 6, 7,

    //8, 9, 10,
    //8, 10, 11, 

    12, 15, 14,
    12, 14, 13,

    16, 17, 18,
    16, 18, 19,

    20, 23, 22,
    20, 22, 21
  };

  SimpleMesh cube;
  
  cube.vPos.resize(numberVertices * 4);
  cube.vNorm.resize(numberVertices * 4);
  cube.vTexCoord.resize(numberVertices * 2);
  cube.triIndices.resize(numberIndices);
  cube.matIndices.resize(numberIndices / 3);
  
  for (size_t i = 0; i < cube.vPos.size(); i++)
    cube.vPos[i] = cubeVertices[i] * a_size;
  
  for (size_t i = 0; i < cube.vNorm.size(); i++)
    cube.vNorm[i] = cubeNormals[i]*(-1.0f);
  
  for (size_t i = 0; i < cube.vTexCoord.size(); i++)
    cube.vTexCoord[i] = cubeTexCoords[i];
  
  for (size_t i = 0; i < cube.triIndices.size(); i++)
    cube.triIndices[i] = cubeIndices[i];
  
  for (size_t i = 0; i < cube.matIndices.size(); i++)
    cube.matIndices[i] = 0;
  
  return cube;
}

// void crossProduct(float result[3], const float vector0[3], const float vector1[3])
// {
//   result[0] = vector0[1] * vector1[2] - vector0[2] * vector1[1];
//   result[1] = vector0[2] * vector1[0] - vector0[0] * vector1[2];
//   result[2] = vector0[0] * vector1[1] - vector0[1] * vector1[0];
// }

SimpleMesh CreateSphere(float radius, int numberSlices)
{
  SimpleMesh sphere;

  int i, j;

  int numberParallels = numberSlices;
  int numberVertices  = (numberParallels + 1) * (numberSlices + 1);
  int numberIndices   = numberParallels * numberSlices * 3;

  float angleStep     = (2.0f * 3.14159265358979323846f) / ((float)numberSlices);
  float helpVector[3] = { 0.0f, 1.0f, 0.0f };
 
  sphere.vPos.resize(numberVertices * 4);
  sphere.vNorm.resize(numberVertices * 4);
  sphere.vTexCoord.resize(numberVertices * 2);

  sphere.triIndices.resize(numberIndices);
  sphere.matIndices.resize(numberIndices/3);

  for (i = 0; i < numberParallels + 1; i++)
  {
    for (j = 0; j < numberSlices + 1; j++)
    {
      int vertexIndex    = (i * (numberSlices + 1) + j) * 4;
      int normalIndex    = (i * (numberSlices + 1) + j) * 4;
      int texCoordsIndex = (i * (numberSlices + 1) + j) * 2;

      sphere.vPos[vertexIndex + 0] = radius * sinf(angleStep * (float)i) * sinf(angleStep * (float)j);
      sphere.vPos[vertexIndex + 1] = radius * cosf(angleStep * (float)i);
      sphere.vPos[vertexIndex + 2] = radius * sinf(angleStep * (float)i) * cosf(angleStep * (float)j);
      sphere.vPos[vertexIndex + 3] = 1.0f;

      sphere.vNorm[normalIndex + 0] = sphere.vPos[vertexIndex + 0] / radius;
      sphere.vNorm[normalIndex + 1] = sphere.vPos[vertexIndex + 1] / radius;
      sphere.vNorm[normalIndex + 2] = sphere.vPos[vertexIndex + 2] / radius;
      sphere.vNorm[normalIndex + 3] = 1.0f;

      sphere.vTexCoord[texCoordsIndex + 0] = (float)j / (float)numberSlices;
      sphere.vTexCoord[texCoordsIndex + 1] = (1.0f - (float)i) / (float)(numberParallels - 1);
    }
  }

  int* indexBuf = &sphere.triIndices[0];

  for (i = 0; i < numberParallels; i++)
  {
    for (j = 0; j < numberSlices; j++)
    {
      *indexBuf++ = i * (numberSlices + 1) + j;
      *indexBuf++ = (i + 1) * (numberSlices + 1) + j;
      *indexBuf++ = (i + 1) * (numberSlices + 1) + (j + 1);

      *indexBuf++ = i * (numberSlices + 1) + j;
      *indexBuf++ = (i + 1) * (numberSlices + 1) + (j + 1);
      *indexBuf++ = i * (numberSlices + 1) + (j + 1);
      
      int diff = int(indexBuf - &sphere.triIndices[0]);
      if (diff >= numberIndices)
        break;
    }

    int diff = int(indexBuf - &sphere.triIndices[0]);
    if (diff >= numberIndices)
      break;
  }

  //int diff = indexBuf - &sphere.triIndices[0];

  return sphere;
}



SimpleMesh CreateTorus(float innerRadius, float outerRadius, int numSides, int numFaces)
{
  SimpleMesh torus;

  // t, s = parametric values of the equations, in the range [0,1]
  float t = 0;
  float s = 0;

  // incr_t, incr_s are increment values aplied to t and s on each loop iteration	to generate 
  float tIncr;
  float sIncr;

  // to store precomputed sin and cos values 
  float cos2PIt, sin2PIt, cos2PIs, sin2PIs;

  int numberVertices;
  int numberIndices;

  // used later to help us calculating tangents vectors
  float helpVector[3] = { 0.0f, 1.0f, 0.0f };

  // indices for each type of buffer (of vertices, indices, normals...)
  int indexVertices, indexIndices, indexNormals, indexTexCoords;

  // loop counters
  int sideCount, faceCount;

  // used to generate the indices
  int v0, v1, v2, v3;

  numberVertices = (numFaces + 1) * (numSides + 1);
  numberIndices  = numFaces * numSides * 2 * 3; // 2 triangles per face * 3 indices per triang

  torus.vPos.resize(numberVertices * 4);
  torus.vNorm.resize(numberVertices * 4);
  torus.vTexCoord.resize(numberVertices * 2);
  torus.triIndices.resize(numberIndices);
  torus.matIndices.resize(numberIndices / 3);
  
  tIncr = 1.0f / (float)numFaces;
  sIncr = 1.0f / (float)numSides;

  // generate vertices and its attributes
  for (sideCount = 0; sideCount <= numSides; ++sideCount, s += sIncr)
  {
    // precompute some values
    cos2PIs = (float)cos(2.0f*3.14159265358979323846f*s);
    sin2PIs = (float)sin(2.0f*3.14159265358979323846f*s);

    t = 0.0f;
    for (faceCount = 0; faceCount <= numFaces; ++faceCount, t += tIncr)
    {
      // precompute some values
      cos2PIt = (float)cos(2.0f*3.14159265358979323846f*t);
      sin2PIt = (float)sin(2.0f*3.14159265358979323846f*t);

      // generate vertex and stores it in the right position
      indexVertices = ((sideCount * (numFaces + 1)) + faceCount) * 4;
      torus.vPos[indexVertices + 0] = (outerRadius + innerRadius * cos2PIt) * cos2PIs;
      torus.vPos[indexVertices + 1] = (outerRadius + innerRadius * cos2PIt) * sin2PIs;
      torus.vPos[indexVertices + 2] = innerRadius * sin2PIt;
      torus.vPos[indexVertices + 3] = 1.0f;

      // generate normal and stores it in the right position
      // NOTE: cos (2PIx) = cos (x) and sin (2PIx) = sin (x) so, we can use this formula
      // ormal = {cos(2PIs)cos(2PIt) , sin(2PIs)cos(2PIt) ,sin(2PIt)}      
      indexNormals = ((sideCount * (numFaces + 1)) + faceCount) * 4;
      torus.vNorm[indexNormals + 0] = cos2PIs * cos2PIt;
      torus.vNorm[indexNormals + 1] = sin2PIs * cos2PIt;
      torus.vNorm[indexNormals + 2] = sin2PIt;
      torus.vNorm[indexNormals + 3] = 0.0f;

      // generate texture coordinates and stores it in the right position
      indexTexCoords = ((sideCount * (numFaces + 1)) + faceCount) * 2;
      torus.vTexCoord[indexTexCoords + 0] = t;
      torus.vTexCoord[indexTexCoords + 1] = s;

    }
  }


  indexIndices = 0;
  for (sideCount = 0; sideCount < numSides; ++sideCount)
  {
    for (faceCount = 0; faceCount < numFaces; ++faceCount)
    {
      // get the number of the vertices for a face of the torus. They must be < numVertices
      v0 = ((sideCount * (numFaces + 1)) + faceCount);
      v1 = (((sideCount + 1) * (numFaces + 1)) + faceCount);
      v2 = (((sideCount + 1) * (numFaces + 1)) + (faceCount + 1));
      v3 = ((sideCount * (numFaces + 1)) + (faceCount + 1));

      // first triangle of the face, counter clock wise winding		
      torus.triIndices[indexIndices++] = v0;
      torus.triIndices[indexIndices++] = v1;
      torus.triIndices[indexIndices++] = v2;

      // second triangle of the face, counter clock wise winding
      torus.triIndices[indexIndices++] = v0;
      torus.triIndices[indexIndices++] = v2;
      torus.triIndices[indexIndices++] = v3;
    }
  }

  return torus;
}
