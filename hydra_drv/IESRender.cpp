// � Copyright 2017 Vladimir Frolov, MSU Grapics & Media Lab

#include <vector>
#include <iostream>
#include <cstring>
#include <cmath>

#ifndef M_PI
static const float M_PI       = 3.14159265358979323846f;
#endif

#ifndef INV_PI
static const float INV_PI     = 1.0f/M_PI;
#endif

#ifndef DEG_TO_RAD
static const float DEG_TO_RAD = M_PI / 180.0f;
#endif

namespace oldies
{
  extern "C" {
  #include "../../HydraAPI/ies_parser/IESNA.H"
  };
};

enum IES_REFLECT { REFLECT4, REFLECT2, REFLECT0 };

std::vector<float> CreateSphericalTextureFromIES(const std::string& a_iesData, int* pW, int* pH)
{
  oldies::IE_DATA iesOldCrap;
  memset(&iesOldCrap, 0, sizeof(oldies::IE_DATA));
  bool read = (oldies::IE_ReadFile((char*)a_iesData.c_str(), &iesOldCrap) != 0);

  if (!read || iesOldCrap.photo.vert_angles == nullptr)
  {
    std::cerr << "oldies::IE_ReadFile error" << std::endl;
    (*pW) = 1;
    (*pH) = 1;
    return std::vector<float>(1);
  }

  int w = 0; // phi, 0-360
  int h = 0; // theta, 0-180

  float verticalStart = iesOldCrap.photo.vert_angles[0];
  float verticalEnd   = iesOldCrap.photo.vert_angles[iesOldCrap.photo.num_vert_angles - 1]; // 0-90, 90-180, 0-180

  float horizontStart = iesOldCrap.photo.horz_angles[0];
  float horizontEnd   = iesOldCrap.photo.horz_angles[iesOldCrap.photo.num_horz_angles - 1];   // 0-0, 0-90, 0-180, 180-360, 0-360

  float eps = 1e-5f;

  //
  //
  if (fabs(verticalStart) < eps && fabs(verticalEnd - 90.0f) < eps) // 0-90
    h = iesOldCrap.photo.num_vert_angles * 2;
  else if (fabs(verticalStart - 90.0f) < eps && fabs(verticalEnd - 180.0f) < eps) // 90-180
    h = iesOldCrap.photo.num_vert_angles * 2;
  else // 0-180
    h = iesOldCrap.photo.num_vert_angles;

  IES_REFLECT reflectType = REFLECT0;

  // The set of horizontal angles, listed in increasing order.The first angle must be 0�.
  // The last angle determines the degree of lateral symmetry displayed by the intensity distribution.If it is 0�, the distribution is axially symmetric. 
  // If it is 90�, the distribution is symmetric in each quadrant.
  // If it is 180�, the distribution is symmetric about a vertical plane.
  // If it is greater than 180� and less than or equal to 360�, the distribution exhibits no lateral symmetries.
  // All other values are invalid.
  //
  if (fabs(horizontStart) < eps && fabs(horizontEnd) < eps) // 0-0
    w = 1;
  else if (fabs(horizontStart) < eps && fabs(horizontEnd - 90.0f) < eps) // 0-90, If it is 90�, the distribution is symmetric in each quadrant.
  {
    w = iesOldCrap.photo.num_horz_angles * 4;
    reflectType = REFLECT4;
  }
  else if (fabs(horizontStart) < eps && fabs(horizontEnd - 180.0f) < eps) // 0-180,  If it is 180�, the distribution is symmetric about a vertical plane.
  {
    w = iesOldCrap.photo.num_horz_angles * 4;
    reflectType = REFLECT4;
  }
  else if (fabs(horizontStart - 90.0f) < eps && fabs(horizontEnd - 180.0f) < eps) // 90-180
  {
    w = iesOldCrap.photo.num_horz_angles * 2;
    reflectType = REFLECT2;
  }
  else // if (fabs(horizontStart) < eps && fabs(horizontEnd - 360.0f) < eps) // 0-360
    w = iesOldCrap.photo.num_horz_angles;

  if (horizontEnd > 180.0f && horizontEnd < 360.0f) // crappy values like 357.0
    horizontEnd = 360.0f;


  std::vector<float> resultData(w*h);
  for (auto i = 0; i < resultData.size(); i++)
    resultData[i] = 0.0f;

  // init initial data
  //
  float stepTheta = (verticalEnd - verticalStart) / float(iesOldCrap.photo.num_vert_angles);
  float stepPhi   = (horizontEnd - horizontStart) / float(iesOldCrap.photo.num_horz_angles);
  if (fabs(stepPhi) < eps)
    stepPhi = 360.0f;

  int thetaIndex = 0;
  for (float thetaGrad = verticalStart; (thetaIndex < iesOldCrap.photo.num_vert_angles); thetaGrad += stepTheta, thetaIndex++)
  {
    float theta = DEG_TO_RAD*thetaGrad;
    int iY = int((theta*INV_PI)*float(h) + 0.5f);
    if (iY >= h)
      iY = h - 1;

    int phiIndex = 0;
    for (float phiGrad = horizontStart; (phiIndex < iesOldCrap.photo.num_horz_angles); phiGrad += stepPhi, phiIndex++)
    {
      float phi = DEG_TO_RAD*phiGrad;
      int   iX = int((phi*0.5f*INV_PI)*float(w) + 0.5f);
      if (iX >= w)
        iX = w - 1;

      resultData[iY*w + iX] = iesOldCrap.photo.pcandela[phiIndex][thetaIndex];
    }
  }

  // reflect the rest of surrounding sphere
  //
  if (reflectType == REFLECT4)
  {
    for (float thetaGrad = verticalStart; (thetaGrad <= verticalEnd); thetaGrad += stepTheta)
    {
      float theta = DEG_TO_RAD*thetaGrad;
      int iY = int((theta*INV_PI)*float(h) + 0.5f);
      if (iY >= h)
        iY = h - 1;

      for (float phiGrad = 0.0f; (phiGrad <= 90.0f); phiGrad += stepPhi)
      {
        float phi  = DEG_TO_RAD*(phiGrad);
        float phi2 = DEG_TO_RAD*(180.0f - phiGrad - stepPhi);
        float phi3 = DEG_TO_RAD*(180.0f + phiGrad);
        float phi4 = DEG_TO_RAD*(360.0f - phiGrad - stepPhi);

        int iX1 = int((phi*0.5f*INV_PI)*float(w) + 0.5f);
        int iX2 = int((phi2*0.5f*INV_PI)*float(w) + 0.5f);
        int iX3 = int((phi3*0.5f*INV_PI)*float(w) + 0.5f);
        int iX4 = int((phi4*0.5f*INV_PI)*float(w) + 0.5f);

        if (iX1 >= w) iX1 = w - 1;
        if (iX2 >= w) iX2 = w - 1;
        if (iX3 >= w) iX3 = w - 1;
        if (iX4 >= w) iX4 = w - 1;

        resultData[iY*w + iX2] = resultData[iY*w + iX1];
        resultData[iY*w + iX3] = resultData[iY*w + iX1];
        resultData[iY*w + iX4] = resultData[iY*w + iX1];
      }
    }
  }
  else if (reflectType == REFLECT2)
  {

    for (float thetaGrad = verticalStart; (thetaGrad <= verticalEnd); thetaGrad += stepTheta)
    {
      float theta = DEG_TO_RAD*thetaGrad;
      int iY = int((theta*INV_PI)*float(h) + 0.5f);
      if (iY >= h)
        iY = h - 1;

      for (float phiGrad = 0.0f; (phiGrad <= 180.0f); phiGrad += stepPhi)
      {
        float phi  = DEG_TO_RAD*phiGrad;
        float phi2 = DEG_TO_RAD*(360.0f - phiGrad - stepPhi);

        int iX1 = int((phi*0.5f*INV_PI)*float(w) + 0.5f);
        int iX2 = int((phi2*0.5f*INV_PI)*float(w) + 0.5f);

        if (iX1 >= w) iX1 = w - 1;
        if (iX2 >= w) iX2 = w - 1;

        resultData[iY*w + iX2] = resultData[iY*w + iX1];
      }
    }
  }

  if (pW != nullptr) (*pW) = w;
  if (pH != nullptr) (*pH) = h;

  oldies::IE_Flush(&iesOldCrap); // release resources

  // auto maxVal = *(std::max_element(resultData.begin(), resultData.end()));
  // auto minVal = *(std::min_element(resultData.begin(), resultData.end()));
  // 
  // //normalize intensity values to control it later with multiplier
  // std::for_each(resultData.begin(), resultData.end(), [=](float &x) { x = (x - minVal) / (maxVal - minVal); });

  return resultData;
}

