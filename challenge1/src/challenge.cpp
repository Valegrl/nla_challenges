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

  /* 1) ******************************************************************************************
    Load the image as an Eigen matrix with size m × n. Each entry in the matrix corresponds
    to a pixel on the screen and takes a value somewhere between 0 (black) and 255 (white).
    Report the size of the matrix.
  */

  // Load the image using stb_image
  int width, height, channels;
  // for greyscale images force to load only one channel
  unsigned char* image_data = stbi_load(input_image_path, &width, &height, &channels, 1);
  if (!image_data) {
    std::cerr << "Error: Could not load image " << input_image_path << std::endl;
    return 1;
  }

  // Print the image size
  std::cout << "1) Image loaded: " << width << "x" << height << " with "
            << channels << " channels.\n" << std::endl;

  /* 2) ******************************************************************************************
    Introduce a noise signal into the loaded image by adding random fluctuations of color
    ranging between [−50, 50] to each pixel. Export the resulting image in .png and upload it.
  */

  // Prepare the matrices to store the image data
  MatrixXd original(height, width), noise(height, width);

  MatrixXd gaussian_noise = MatrixXd::Random(height, width);
  // Fill the matrices with image data and create the noised image using the gaussian noise
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int index = (i * width + j) * channels;  // 1 channel (Greyscale) 3 channels (RGB)
      original(i, j) = static_cast<double>(image_data[index]) / 255;
      // make a random int from -50 to 50
      double noise_value = static_cast<double>(image_data[index]) + static_cast<double>(gaussian_noise(i, j)) * 50;
      // Clamp the value between 0 and 255
      noise(i, j) = std::clamp(noise_value, 0.0, 255.0) / 255;
    }
  }
  // Free memory!!!
  stbi_image_free(image_data);

  // Convert the matrix to a char matrix to save the image
  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> noised_image_char(height, width);
  noised_image_char = noise.unaryExpr([](double val) -> unsigned char {
    return static_cast<unsigned char>(val * 255.0);
  });

  const std::string output_image_path1 = "noised_image.png";

  // Save the image using stb_image_write
  if (stbi_write_png(output_image_path1.c_str(), width, height, 1,
                     noised_image_char.data(), width) == 0) {
    std::cerr << "Error: Could not save noised image" << std::endl;
    return 1;
  }
  std::cout << "2) Noised image saved to " << output_image_path1 << "\n" << std::endl;

  /* 3) ******************************************************************************************
    Reshape the original and noisy images as vectors v and w, respectively. Verify that each
    vector has m n components. Report here the Euclidean norm of v.
  */

  // Create a vector from the matrix
  VectorXd v(width*height), w(width*height);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      w(i*width+j) = noise(i,j);
      v(i*width+j) = original(i,j);
    }
  }

  // Print the size of the vectors and the norm of the vector v
  std::cout << "3) " << std::endl;
  std::cout << "v size: " << v.size() << std::endl;
  std::cout << "w size: " << w.size() << std::endl;
  std::cout << "v norm: " << v.norm() << std::endl;
  std::cout << "\n" << std::endl;


  /* 4) ******************************************************************************************
    Write the convolution operation corresponding to the smoothing kernel Hav2 as a matrix
    vector multiplication between a matrix A1 having size mn × mn and the image vector.
    Report the number of non-zero entries in A1.
  */

  // Create the convolutional matrix
  MatrixXd Av2 = MatrixXd::Constant(3,3,1.0) / 9.0;
  SparseMatrix<double,RowMajor> A1(width*height, width*height);

  // Fill the convulutional sparse matrix to operate the operation A1*image_vector = smoothed_image_vector
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      for (int k = 0; k < 3; ++k) {
        // If an index is out of bounds, treat it as zero
        if (i+k-1 < 0 || i+k-1 >= height) {
          continue;
        }
        for (int l = 0; l < 3; ++l) {
          if (j+l-1 < 0 || j+l-1 >= width) {
            continue;
          }
          A1.insert(i*width+j, (i+k-1)*width+j+l-1) = Av2(k,l);
        }
      }
    }
  }
  // print the number of non zeros
  std::cout << "4) Number of non zeros: " << A1.nonZeros() << "\n" << std::endl;

  /* 5) ******************************************************************************************
    Apply the previous smoothing filter to the noisy image by performing the matrix vector
    multiplication A1w. Export and upload the resulting image.
  */

  //Calculate the finale smoothed image with A1w
  VectorXd smoothed_image = A1*w;

  //Convert vector to the matrix
  MatrixXd smoothed_image_matrix = MatrixXd::Zero(height, width);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      smoothed_image_matrix(i,j) = smoothed_image(i*width+j);
    }
  }

  // Convert the matrix to a char matrix to save the image
  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> smoothed_image_char(height, width);
  smoothed_image_char = smoothed_image_matrix.unaryExpr([](double val) -> unsigned char {
    return static_cast<unsigned char>(val * 255.0);
  });

  const std::string output_image_path2 = "smoothed_image.png";

  // Save the image using stb_image_write
  if (stbi_write_png(output_image_path2.c_str(), width, height, 1,
                     smoothed_image_char.data(), width) == 0) {
    std::cerr << "Error: Could not save smoothed image" << std::endl;
    return 1;
  }
  std::cout << "5) Smoothed image saved to " << output_image_path2 << "\n" << std::endl;

  /* 6) ******************************************************************************************
    Write the convolution operation corresponding to the sharpening kernel Hsh2 as a matrix
    vector multiplication by a matrix A2 having size mn×mn. Report the number of non-zero
    entries in A2. Is A2 symmetric?
  */

  // Create the sharpening matrix
  MatrixXd sh2(3,3);
  sh2(0,0) = 0;
  sh2(0,1) = -3;
  sh2(0,2) = 0;
  sh2(1,0) = -1;
  sh2(1,1) = 9;
  sh2(1,2) = -3;
  sh2(2,0) = 0;
  sh2(2,1) = -1;
  sh2(2,2) = 0;

  SparseMatrix<double,RowMajor> A2(width*height, width*height);

  // Fill the convulutional sparse matrix to operate the operation A2*image_vector = sharpened_image_vector
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      for (int k = 0; k < 3; ++k) {
        // If an index is out of bounds, treat it as zero
        if (i+k-1 < 0 || i+k-1 >= height) {
          continue;
        }
        for (int l = 0; l < 3; ++l) {
          if (j+l-1 < 0 || j+l-1 >= width) {
            continue;
          }
          // Fill the matrix with only the non zero values
          if (sh2(k, l) != 0) {
            A2.insert(i * width + j, (i + k - 1) * width + j + l - 1) = sh2(k, l);
          }
        }
      }
    }
  }

  // print the number of non zeros
  std::cout << "6) Number of non zeros: " << A2.nonZeros() << "\n" << std::endl;

  // Check if A2 is symmetric
  bool is_symmetric = A2.isApprox(A2.transpose());
  std::cout << "Is A2 symmetric? " << (is_symmetric ? "Yes" : "No") << "\n" << std::endl;

  /* 7) ******************************************************************************************
    Apply the previous sharpening filter to the original image by performing the matrix vector
    multiplication A2v. Export and upload the resulting image.
  */

  // Calculate the final sharpened image with A2*v
  VectorXd sharpened_image = A2 * (v * 255.0);

  // Convert vector to the matrix and clamp the values between 0 and 255
  MatrixXd sharpened_image_matrix = MatrixXd::Zero(height, width);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int index = i * width + j;
      double value = std::clamp(sharpened_image(index), 0.0, 255.0);
      sharpened_image_matrix(i, j) = value;
    }
  }

  // Convert the matrix to a char matrix to save the image
  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> sharpened_image_char(height, width);
  sharpened_image_char = sharpened_image_matrix.unaryExpr([](double val) -> unsigned char {
    return static_cast<unsigned char>(val);
  });

  const std::string output_image_path3 = "sharpened_image.png";

  // Save the image using stb_image_write
  if (stbi_write_png(output_image_path3.c_str(), width, height, 1,
                     sharpened_image_char.data(), width) == 0) {
    std::cerr << "Error: Could not save sharpened image" << std::endl;
    return 1;
  }
  std::cout << "7) Sharpened image saved to " << output_image_path3 << "\n" << std::endl;

  /* 8) ******************************************************************************************
    Export the Eigen matrix A2 and vector w in the .mtx format. Using a suitable iterative
    solver and preconditioner technique available in the LIS library compute the approximate
    solution to the linear system A2x = w prescribing a tolerance of 10−9. Report here the
    iteration count and the final residual.
  */

  // TODO

  /* 9) ******************************************************************************************
    Import the previous approximate solution vector x in Eigen and then convert it into a .png
    image. Upload the resulting file here.
  */

  // TODO
  
  /* 10) *****************************************************************************************
    Write the convolution operation corresponding to the detection kernel Hlap as a matrix
    vector multiplication by a matrix A3 having size mn × mn. Is matrix A3 symmetric?
  */

  // Create the laplacian matrix
  MatrixXd lap(3,3);
  lap(0,0) = 0;
  lap(0,1) = -1;
  lap(0,2) = 0;
  lap(1,0) = -1;
  lap(1,1) = 4;
  lap(1,2) = -1;
  lap(2,0) = 0;
  lap(2,1) = -1;
  lap(2,2) = 0;

  SparseMatrix<double,RowMajor> A3(width*height, width*height);

  // Fill the convulutional sparse matrix to operate the operation A3*image_vector = detected_image_vector
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      for (int k = 0; k < 3; ++k) {
        // If an index is out of bounds, treat it as zero
        if (i+k-1 < 0 || i+k-1 >= height) {
          continue;
        }
        for (int l = 0; l < 3; ++l) {
          if (j+l-1 < 0 || j+l-1 >= width) {
            continue;
          }
          // Fill the matrix with only the non zero values
          if (lap(k, l) != 0) {
            A3.insert(i * width + j, (i + k - 1) * width + j + l - 1) = lap(k, l);
          }
        }
      }
    }
  }

  // Check if A3 is symmetric
  bool is_symmetric_A3 = A3.isApprox(A3.transpose());
  std::cout << "10) Is A3 symmetric? " << (is_symmetric_A3 ? "Yes" : "No") << "\n" << std::endl;

  /* 11) *****************************************************************************************
    Apply the previous edge detection filter to the original image by performing the matrix
    vector multiplication A3 v. Export and upload the resulting image.
  */

  // Calculate the final edge detected image with A3*v
  VectorXd edge_detected_image = A3 * (v * 255.0);

  // Convert vector to the matrix and clamp the values between 0 and 255
  MatrixXd edge_detected_image_matrix = MatrixXd::Zero(height, width);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int index = i * width + j;
      double value = std::clamp(edge_detected_image(index), 0.0, 255.0);
      edge_detected_image_matrix(i, j) = value;
    }
  }

  // Convert the matrix to a char matrix to save the image
  Matrix<unsigned char, Dynamic, Dynamic, RowMajor> edge_detected_image_char(height, width);
  edge_detected_image_char = edge_detected_image_matrix.unaryExpr([](double val) -> unsigned char {
    return static_cast<unsigned char>(val);
  });

  const std::string output_image_path4 = "edge_detected_image.png";

  // Save the image using stb_image_write
  if (stbi_write_png(output_image_path4.c_str(), width, height, 1,
                     edge_detected_image_char.data(), width) == 0) {
    std::cerr << "Error: Could not save edge detected image" << std::endl;
    return 1;
  }

  std::cout << "11) Edge detected image saved to " << output_image_path4 << "\n" << std::endl;

  /* 12) *****************************************************************************************
    Using a suitable iterative solver available in the Eigen library compute the approximate
    solution of the linear system (I+A3)y = w, where I denotes the identity matrix, prescribing
    a tolerance of 10−10. Report here the iteration count and the final residual.
  */

  // TODO

  /* 13) *****************************************************************************************
    Convert the image stored in the vector y into a .png image and upload it.
  */

  // TODO

  /* 14) *****************************************************************************************
    Comment the obtained results.
  */

  // TODO

  return 0;
}