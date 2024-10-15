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
    std::cout << "\n1) Image loaded: " << width << "x" << height << " with "
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

    // Print the last two positions. 
    // Remember that the eigenvalues are sorted in ascending order using SelfAdjointEigenSolver
    std::cout << "2) The two largest singular values of ATA are: " << std::endl;
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

    std::cout << "3) (Yes: 1.608332e+04) Check with lis library\n" << std::endl;

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

   std::cout << "4) (-shift 200.0) Check with lis library" << std::endl;
   std::cout << "number of iterations = 7 (instead of 8)" << std::endl;
   std::cout << "elapsed time  = 2.114560e-04 sec.\n" << std::endl;

    /* 5) ******************************************************************************************
        Using the SVD module of the Eigen library, perform a singular value decomposition of the
        matrix A. Report the Euclidean norm of the diagonal matrix Σ of the singular values.
    */

    /* 6)  ******************************************************************************************
        Compute the matrices C and D described in (1) assuming k = 40 and k = 80. Report the
        number of nonzero entries in the matrices C and D.
    */

    /* 7) ******************************************************************************************
        Compute the compressed images as the matrix product CDT (again for k = 40 and k = 80).
        Export and upload the resulting images in .png.
    */
   
    /* 8) ******************************************************************************************
        Using Eigen create a black and white checkerboard image (as the one depicted below)
        with height and width equal to 200 pixels. Report the Euclidean norm of the matrix
        corresponding to the image.
    */

    /* 9) ******************************************************************************************
        Introduce a noise into the checkerboard image by adding random fluctuations of color
        ranging between [−50, 50] to each pixel. Export the resulting image in .png and upload it.
    */

    /* 10) *****************************************************************************************
        Using the SVD module of the Eigen library, perform a singular value decomposition of the
        matrix corresponding to the noisy image. Report the two largest computed singular values.
    */

    /* 11) *****************************************************************************************
        Starting from the previously computed SVD, create the matrices C and D defined in (1)
        assuming k = 5 and k = 10. Report the size of the matrices C and D.
    */

    /* 12) *****************************************************************************************
        Compute the compressed images as the matrix product CDT (again for k = 5 and k = 10).
        Export and upload the resulting images in .png
    */

    /* 13) *****************************************************************************************
        Compare the compressed images with the original and noisy images. Comment the results
    */

    return 0;
}