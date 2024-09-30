#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <Eigen/Sparse>

// from https://github.com/nothings/stb/tree/master
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace Eigen;

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
    return 1;
  }

  const char* input_image_path = argv[1];

  // Load the image using stb_image
  int width, height, channels;
  // for greyscale images force to load only one channel
  unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 1);
  if (!image_data) {
    std::cerr << "Error: Could not load image " << input_image_path
              << std::endl;
    return 1;
  }

  std::cout << "Image loaded: " << width << "x" << height << " with "
            << channels << " channels." << std::endl;

  // Prepare Eigen matrices for each RGB channel
  MatrixXd noise(height, width), original(height, width), rotate(width, height);

  MatrixXd gaussian_noise = MatrixXd::Random(height, width);

  // Fill the matrices with image data
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int index = (i * width + j) * channels;  // 1 channel (Greyscale) 3 channels (RGB)
      // make a random int from -50 to 50
      noise(i, j) = std::max(0.0, std::min(255.0, static_cast<double>(image_data[index]) + static_cast<double>(gaussian_noise(i, j)) * 50)) / 255;
      original(i, j) = static_cast<double>(image_data[index]) / 255;
    }
  }
  // Free memory!!!
  stbi_image_free(image_data);

  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> dark_image(height, width);
  // Use Eigen's unaryExpr to map the grayscale values (0.0 to 1.0) to 0 to 255
  dark_image = noise.unaryExpr([](double val) -> unsigned char {
    return static_cast<unsigned char>(val * 255.0);
  });

  VectorXd w(width*height), v(width*height);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      w(i*width+j) = noise(i,j);
      v(i*width+j) = original(i,j);
    }
  }
  std::cout << "v size: " << v.size() << std::endl;
  std::cout << "w size: " << w.size() << std::endl;
  std::cout << "v norm: " << v.norm() << std::endl;

  // Save the image using stb_image_write
  const std::string output_image_path1 = "noised_image.png";
  if (stbi_write_png(output_image_path1.c_str(), width, height, 1,
                     dark_image.data(), width) == 0) {
    std::cerr << "Error: Could not save grayscale image" << std::endl;

    return 1;
  }

  std::cout << "Images saved to " << output_image_path1 << std::endl;

  int A1_rows = v.size();
  int A1_cols = A1_rows;
  int s = 3;

  SparseMatrix<double, RowMajor> A1(A1_rows,A1_cols);
  for (int k=0; k < A1_rows; k++) {
    for (int j=0; j < s; j++){
      for (int i=0; i < s; i++){
        A1.coeffRef(k, (k + j*width + i) % A1_cols) = 1.0/9;
      }
    }
  }

  std::cout << A1.coeff(0, 0) << " ";
  std::cout << A1.coeff(0, 1) << " ";
  std::cout << A1.coeff(0, 2) << " ";
  std::cout << A1.coeff(0, 4) << " ";
  std::cout << A1.coeff(1, 1) << " ";
  std::cout << A1.coeff(1, 0) << " ";


  // Print the first row
  /*
    for (int col = 0; col < A1_cols; col++) {
        std::cout << A1.coeff(0, col) << " ";
    }
    std::cout << std::endl;
  */

  return 0;
}