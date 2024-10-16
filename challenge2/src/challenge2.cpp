#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/IterativeLinearSolvers>

#include "../lib/lis.h"

// from https://github.com/nothings/stb/tree/master
#define STB_IMAGE_IMPLEMENTATION
#include "../lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../lib/stb_image_write.h"

using namespace Eigen;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    const char* input_image_path = argv[1];

    /* 1) ******************************************************************************************
        Load the image as an Eigen matrix A with size m×n. Each entry in the matrix corresponds
        to a pixel on the screen and takes a value somewhere between 0 (black) and 255 (white).
        Compute the matrix product ATA and report the euclidean norm of ATA
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
    std::cout << "\n1)\nImage loaded: " << width << "x" << height << " with "
                << channels << " channels." << std::endl;

    // Prepare the matrices to store the image data
    MatrixXd A(height, width);

    // Fill the matrices with image data
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
        int index = (i * width + j) * channels;  // 1 channel (Greyscale) 3 channels (RGB)
        A(i, j) = static_cast<double>(image_data[index]) / 255;
        }
    }

    // Free memory!!!
    stbi_image_free(image_data);

    // Compute ATA
    MatrixXd ATA = A.transpose() * A;

    // Compute the Euclidean norm of ATA
    double norm = ATA.norm();

    // Print the norm
    std::cout << "norm of A.T * A = " << norm << std::endl;

    // Compute ATA_255
    MatrixXd A_255 = 255 * A;
    MatrixXd ATA_255 = A_255.transpose() * A_255;

    // Compute the Euclidean norm of ATA
    double norm_255 = ATA_255.norm();

    // Print the norm
    std::cout << "(255) norm of A.T * A = " << norm_255 << "\n" << std::endl;

    /* 2) ******************************************************************************************
        Solve the eigenvalue problem ATAx = λx using the proper solver provided by the Eigen
        library. Report the two largest computed singular values of A.
    */

    // Compute the eigenvalues of ATA
    SelfAdjointEigenSolver<MatrixXd> eigensolver(ATA);
    if (eigensolver.info() != Eigen::Success) abort();

    // Get the eigenvalues and save them in a vector
    VectorXd eigenvalues = eigensolver.eigenvalues();

    std::cout << eigenvalues << std::endl;

    // Print the last two positions. 
    // Remember that the eigenvalues are sorted in ascending order using SelfAdjointEigenSolver
    std::cout << "2)\nThe two largest singular values of ATA are: " << std::endl;
    std::cout << eigenvalues(eigenvalues.size() - 2) << std::endl;
    std::cout << eigenvalues(eigenvalues.size() - 1) << "\n" << std::endl;

    /* 3) ******************************************************************************************
        Export matrix ATA in the matrix market format and move it to the lis-2.1.6/test
        folder. Using the proper iterative solver available in the LIS library compute the largest
        eigenvalue of ATA up to a tolerance of 10−8. Report the computed eigenvalue. Is the result
        in agreement with the one obtained in the previous point?
    */

    std::string matrixATAFileOut("ATA.mtx");
    Eigen::saveMarket(ATA, matrixATAFileOut);

    /*
    mpirun -n 4 ./eigen1 ATA.mtx eigvec.mtx hist.txt -e pi -etol 1.0e-8

    number of processes = 4
    matrix size = 256 x 256 (65536 nonzero entries)

    initial vector x      : all components set to 1
    precision             : double
    eigensolver           : Power
    convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-08 * ||lx||_2
    matrix storage format : CSR
    shift                 : 0.000000e+00
    eigensolver status    : normal end

    Power: mode number          = 0
    Power: eigenvalue           = 1.608332e+04
    Power: number of iterations = 8
    Power: elapsed time         = 2.608200e-04 sec.
    Power:   preconditioner     = 0.000000e+00 sec.
    Power:     matrix creation  = 0.000000e+00 sec.
    Power:   linear solver      = 0.000000e+00 sec.
    Power: relative residual    = 1.866013e-09
    */

    std::cout << "3)\n(Yes: 1.608332e+04) Check with lis library\n" << std::endl;

    /* 4) ******************************************************************************************
        Find a shift µ ∈ R yielding an acceleration of the previous eigensolver. Report µ and the
        number of iterations required to achieve a tolerance of 10−8.
    */

    /*
    mpirun -n 4 ./eigen1 ATA.mtx eigvec.mtx hist.txt -e pi -etol 1.0e-8 -shift 200.0

    number of processes = 4
    matrix size = 256 x 256 (65536 nonzero entries)

    initial vector x      : all components set to 1
    precision             : double
    eigensolver           : Power
    convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-08 * ||lx||_2
    matrix storage format : CSR
    shift                 : 2.000000e+02
    eigensolver status    : normal end

    Power: mode number          = 0
    Power: eigenvalue           = 1.608332e+04
    Power: number of iterations = 7
    Power: elapsed time         = 2.114560e-04 sec.
    Power:   preconditioner     = 0.000000e+00 sec.
    Power:     matrix creation  = 0.000000e+00 sec.
    Power:   linear solver      = 0.000000e+00 sec.
    Power: relative residual    = 9.279366e-09
    */

   std::cout << "4)\n(-shift 200.0) Check with lis library" << std::endl;
   std::cout << "number of iterations = 7 (instead of 8)" << std::endl;
   std::cout << "elapsed time  = 2.114560e-04 sec.\n" << std::endl;

    /* 5) ******************************************************************************************
        Using the SVD module of the Eigen library, perform a singular value decomposition of the
        matrix A. Report the Euclidean norm of the diagonal matrix Σ of the singular values.
    */

    // Compute the SVD of A
    BDCSVD<MatrixXd> svd(A, ComputeThinU | ComputeThinV);

    // Get the singular values and save them in a vector
    VectorXd singular_values = svd.singularValues();

    // Compute the Euclidean norm of the singular values
    double singular_values_norm = singular_values.norm();

    // Print the norm
    std::cout << "5)\nEuclidean norm of the singular values: " << singular_values_norm << "\n" << std::endl;

    /* 6)  ******************************************************************************************
        Compute the matrices C and D described in (1) assuming k = 40 and k = 80. Report the
        number of nonzero entries in the matrices C and D.
    */

    // Compute the matrices C and D for k = 40 and k = 80
    int k1 = 40;
    int k2 = 80;

    MatrixXd C_40 = svd.matrixU().leftCols(k1);
    MatrixXd D_40 = svd.matrixV().leftCols(k1) * svd.singularValues().head(k1).asDiagonal();

    // Compute the number of non-zero entries in the matrices C and D
    int nnz_C = C_40.nonZeros();
    int nnz_D = D_40.nonZeros();

    // Print the number of non-zero entries
    std::cout << "6)\nNumber of non-zero entries in C for k = 40: " << nnz_C << std::endl;
    std::cout << "Number of non-zero entries in D for k = 40: " << nnz_D << std::endl;

    // Compute the matrices C and D for k = 80
    MatrixXd C_80 = svd.matrixU().leftCols(k2);
    MatrixXd D_80 = svd.matrixV().leftCols(k2) * svd.singularValues().head(k2).asDiagonal();

    // Compute the number of non-zero entries in the matrices C and D
    nnz_C = C_80.nonZeros();
    nnz_D = D_80.nonZeros();

    // Print the number of non-zero entries
    std::cout << "Number of non-zero entries in C for k = 80: " << nnz_C << std::endl;
    std::cout << "Number of non-zero entries in D for k = 80: " << nnz_D << "\n" << std::endl;
    
    /* 7) ******************************************************************************************
        Compute the compressed images as the matrix product CDT (again for k = 40 and k = 80).
        Export and upload the resulting images in .png.
    */

    // Compute the compressed images for k = 40 and k = 80
    MatrixXd compressed_image1 = C_40 * D_40.transpose();
    MatrixXd compressed_image2 = C_80 * D_80.transpose();

    // Convert the matrix to a char matrix to save the image
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> compressed_image_char1(height, width);
    compressed_image_char1 = compressed_image1.unaryExpr([](double val) -> unsigned char {
        return static_cast<unsigned char>(std::clamp(val * 255.0, 0.0, 255.0));
    });

    const std::string output_image_path3 = "compressed_image_k40.png";

    // Save the image using stb_image_write
    if (stbi_write_png(output_image_path3.c_str(), width, height, 1,
                        compressed_image_char1.data(), width) == 0) {
        std::cerr << "Error: Could not save " << output_image_path3 <<" image" << std::endl;
        return 1;
    }
    std::cout << "7)\nCompressed image for k = 40 saved to " << output_image_path3 << std::endl;

    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> compressed_image_char2(height, width);
    compressed_image_char2 = compressed_image2.unaryExpr([](double val) -> unsigned char {
        return static_cast<unsigned char>(std::clamp(val * 255.0, 0.0, 255.0));
    });

    const std::string output_image_path4 = "compressed_image_k80.png";

    // Save the image using stb_image_write
    if (stbi_write_png(output_image_path4.c_str(), width, height, 1,
                        compressed_image_char2.data(), width) == 0) {
        std::cerr << "Error: Could not save " << output_image_path4 <<" image" << std::endl;
        return 1;
    }
    std::cout << "Compressed image for k = 80 saved to " << output_image_path4 << "\n" << std::endl;

    /* 8) ******************************************************************************************
        Using Eigen create a black and white checkerboard image (as the one depicted below)
        with height and width equal to 200 pixels. Report the Euclidean norm of the matrix
        corresponding to the image.
    */
    int height_checkerboard = 200;
    int width_checkerboard = 200;
    int square_size = 25;

    MatrixXd checkerboard = MatrixXd::Random(height_checkerboard, width_checkerboard);
    for (int i = 0; i < height_checkerboard; ++i) {
        for (int j = 0; j < width_checkerboard; ++j) {
            checkerboard(i, j) = static_cast<double>((i / square_size % 2 == j / square_size % 2) ? 0.0 : 1.0);
        }
    }

    // Convert the matrix to a char matrix to save the image
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> checkerboard_image_char(height, width);
    checkerboard_image_char = checkerboard.unaryExpr([](double val) -> unsigned char {
        return static_cast<unsigned char>(val * 255.0);
    });

    const std::string output_image_path1 = "original_checkerboard.png";

    // Save the image using stb_image_write
    if (stbi_write_png(output_image_path1.c_str(), width_checkerboard, height_checkerboard, 1,
                        checkerboard_image_char.data(), width_checkerboard) == 0) {
        std::cerr << "Error: Could not save " << output_image_path1 <<" image" << std::endl;
        return 1;
    }
    std::cout << "8)\nOriginal checkerboard image saved to " << output_image_path1 << std::endl;

    // Calculate the Euclidean norm
    double checkerboard_norm = checkerboard.norm();

    std::cout << "Euclidean norm of the checkerboard matrix: " << checkerboard_norm << "\n" << std::endl;

    /* 9) ******************************************************************************************
        Introduce a noise into the checkerboard image by adding random fluctuations of color
        ranging between [−50, 50] to each pixel. Export the resulting image in .png and upload it.
    */

    MatrixXd noised_checkerboard(height_checkerboard, width_checkerboard);

    MatrixXd gaussian_noise = MatrixXd::Random(height_checkerboard, width_checkerboard);
    // Fill the matrices with image data and create the noised image using the gaussian noise
    for (int i = 0; i < height_checkerboard; ++i) {
        for (int j = 0; j < width_checkerboard; ++j) {
        // make a random int from -50 to 50
        double noise_value = static_cast<double>((checkerboard(i, j) * 255.0)) + static_cast<double>(gaussian_noise(i, j)) * 50;
        // Clamp the value between 0 and 255
        noised_checkerboard(i, j) = std::clamp(noise_value, 0.0, 255.0);
        }
    }

    // Convert the matrix to a char matrix to save the image
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> noised_checkerboard_image_char(height, width);
    noised_checkerboard_image_char = noised_checkerboard.unaryExpr([](double val) -> unsigned char {
        return static_cast<unsigned char>(val);
    });

    const std::string output_image_path2 = "noised_checkerboard.png";

    // Save the image using stb_image_write
    if (stbi_write_png(output_image_path2.c_str(), width_checkerboard, height_checkerboard, 1,
                        noised_checkerboard_image_char.data(), width_checkerboard) == 0) {
        std::cerr << "Error: Could not save " << output_image_path2 <<" image" << std::endl;
        return 1;
    }
    std::cout << "9)\nNoised checkerboard image saved to " << output_image_path2 << "\n" << std::endl;

    /* 10) *****************************************************************************************
        Using the SVD module of the Eigen library, perform a singular value decomposition of the
        matrix corresponding to the noisy image. Report the two largest computed singular values.
    */

    // Compute the SVD of the noised checkerboard image
    BDCSVD<MatrixXd> svd_noised_checkerboard(noised_checkerboard, ComputeThinU | ComputeThinV);

    // Get the singular values and save them in a vector
    VectorXd singular_values_noised_checkerboard = svd_noised_checkerboard.singularValues();

    // Print the first two positions.
    std::cout << "10)\nThe two largest singular values of the noised checkerboard image are: " << std::endl;
    std::cout << singular_values_noised_checkerboard(0) << std::endl;
    std::cout << singular_values_noised_checkerboard(1) << "\n" << std::endl;

    /* 11) *****************************************************************************************
        Starting from the previously computed SVD, create the matrices C and D defined in (1)
        assuming k = 5 and k = 10. Report the size of the matrices C and D.
    */

    // Compute the matrices C and D for k = 5 and k = 10
    int k3 = 2;
    int k4 = 10;

    MatrixXd C_5 = svd_noised_checkerboard.matrixU().leftCols(k3);
    MatrixXd D_5 = svd_noised_checkerboard.matrixV().leftCols(k3) * svd_noised_checkerboard.singularValues().head(k3).asDiagonal();

    // Print the size of the matrices C and D
    std::cout << "11)\nSize of the matrices C and D for k = 5: " << std::endl;
    std::cout << "C: " << C_5.rows() << "x" << C_5.cols() << std::endl;
    std::cout << "D: " << D_5.rows() << "x" << D_5.cols() << std::endl;

    // Compute the matrices C and D for k = 10
    MatrixXd C_10 = svd_noised_checkerboard.matrixU().leftCols(k4);
    MatrixXd D_10 = svd_noised_checkerboard.matrixV().leftCols(k4) * svd_noised_checkerboard.singularValues().head(k4).asDiagonal();

    // Print the size of the matrices C and D
    std::cout << "Size of the matrices C and D for k = 10: " << std::endl;
    std::cout << "C: " << C_10.rows() << "x" << C_10.cols() << std::endl;
    std::cout << "D: " << D_10.rows() << "x" << D_10.cols() << "\n" << std::endl;

    /* 12) *****************************************************************************************
        Compute the compressed images as the matrix product CDT (again for k = 5 and k = 10).
        Export and upload the resulting images in .png
    */

    // Compute the compressed images for k = 5 and k = 10
    MatrixXd compressed_image3 = C_5 * D_5.transpose();
    MatrixXd compressed_image4 = C_10 * D_10.transpose();

    // Convert the matrix to a char matrix to save the image
    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> compressed_image_char3(height_checkerboard, width_checkerboard);
    compressed_image_char3 = compressed_image3.unaryExpr([](double val) -> unsigned char {
        return static_cast<unsigned char>(std::clamp(val, 0.0, 255.0));
    });

    const std::string output_image_path5 = "compressed_image_k5.png";

    // Save the image using stb_image_write
    if (stbi_write_png(output_image_path5.c_str(), width_checkerboard, height_checkerboard, 1,
                        compressed_image_char3.data(), width_checkerboard) == 0) {
        std::cerr << "Error: Could not save " << output_image_path5 <<" image" << std::endl;
        return 1;
    }

    std::cout << "12)\nCompressed image for k = 5 saved to " << output_image_path5 << std::endl;

    Matrix<unsigned char, Dynamic, Dynamic, RowMajor> compressed_image_char4(height_checkerboard, width_checkerboard);
    compressed_image_char4 = compressed_image4.unaryExpr([](double val) -> unsigned char {
        return static_cast<unsigned char>(std::clamp(val, 0.0, 255.0));
    });

    const std::string output_image_path6 = "compressed_image_k10.png";

    // Save the image using stb_image_write
    if (stbi_write_png(output_image_path6.c_str(), width_checkerboard, height_checkerboard, 1,
                        compressed_image_char4.data(), width_checkerboard) == 0) {
        std::cerr << "Error: Could not save " << output_image_path6 <<" image" << std::endl;
        return 1;
    }

    std::cout << "Compressed image for k = 10 saved to " << output_image_path6 << "\n" << std::endl;

    /* 13) *****************************************************************************************
        Compare the compressed images with the original and noisy images. Comment the results
    */

    return 0;
}