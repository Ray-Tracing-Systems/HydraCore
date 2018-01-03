#pragma once

#include <vector>

struct HDRImageLite
{
public:

  HDRImageLite();
  HDRImageLite(int w, int h, int channels, const float* data);
  ~HDRImageLite();

  inline int width()  const { return m_width; }
  inline int height() const { return m_height; }
  inline int channels() const { return m_channels; }

  const std::vector<float>& data() const { return m_data; }
  std::vector<float>& data() { return m_data; }

  void gaussBlur(int radius, float sigma);

private:

  int m_width;
  int m_height;
  int m_channels;
  std::vector<float> m_data;

};

void createGaussKernelWeights(int size, std::vector<float>& gKernel, float a_sigma);
void createGaussKernelWeights1D(int size, std::vector<float>& gKernel, float a_sigma);
