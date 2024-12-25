#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdint.h>

// Helper typedef for 2D matrix
typedef std::vector<std::vector<int16_t>> Matrix;

constexpr int image_size = 224;
constexpr int kernel_size = 3;

// Function to print a matrix (for debugging)
void printMatrix(const Matrix& mat) {
    for (const auto& row : mat) {
        for (int16_t val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

// Perform 2D convolution
void convolve(const Matrix& input, const Matrix& kernel, Matrix& output) {
    int inputSize = input.size();
    int kernelSize = kernel.size();
    int outputSize = (input[0].size() - kernelSize) + 1 + 2;

    output.resize(outputSize, std::vector<int16_t>(outputSize, 0));

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            for (int m = 0; m < kernelSize; ++m) {
                for (int n = 0; n < kernelSize; ++n) {
                    if ((i + m) > 0 && (j + n) > 0 && (i + m) < outputSize && (j + n) < outputSize)
                        output[i][j] += input[i + m - 1][j + n - 1] * kernel[m][n];
                }
            }
        }

    }
}
// Apply ReLU activation
void relu(const Matrix& input, Matrix& output) {
    output = input;
    for (auto& row : output) {
        for (auto& val : row) {
            val = std::max(static_cast<short>(0), val);
        }
    }
}

// Upsample by a factor of 2
void upsample(const Matrix& input, Matrix& output) {
    int inputSize = input.size();
    int outputSize = inputSize * 2;
    output.resize(outputSize, std::vector<int16_t>(outputSize, 0));

    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < inputSize; ++j) {
            output[2 * i][2 * j] = input[i][j];
            output[2 * i + 1][2 * j] = input[i][j];
            output[2 * i][2 * j + 1] = input[i][j];
            output[2 * i + 1][2 * j + 1] = input[i][j];
        }
    }
}

// Max pooling with 2x2 filter and stride 2
void maxPool(const Matrix& input, Matrix& output) {
    int inputSize = input.size();
    int outputSize = inputSize / 2;
    output.resize(outputSize, std::vector<int16_t>(outputSize, 0));

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            int16_t maxVal = 0;
            for (int m = 0; m < 2; ++m) {
                for (int n = 0; n < 2; ++n) {
                    maxVal = std::max(maxVal, input[2 * i + m][2 * j + n]);
                }
            }
            output[i][j] = maxVal;
        }
    }
}

void CNN4L(const Matrix& input, const Matrix& weight1, const Matrix& weight2, const Matrix& weight3, const Matrix& weight4, Matrix& output){
    Matrix layer1, layer2, layer3, temp1, temp2, temp3, temp4;
    convolve(input, weight1, temp1);
    relu(temp1, temp1);
    upsample(temp1, layer1);

    std::cout << layer1.size() * layer1[0].size() * 4 << std::endl;

    convolve(layer1, weight2, temp2);
    relu(temp2, temp2);
    upsample(temp2, layer2);

    std::cout << layer2.size() * layer2[0].size() * 4 << std::endl;

    convolve(layer2, weight3, temp3);
    relu(temp3, temp3);
    maxPool(temp3, layer3);

    std::cout << layer3.size() * layer3[0].size() * 4 << std::endl;

    convolve(layer3, weight4, temp4);
    relu(temp4, temp4);
    maxPool(temp4, output);

    std::cout << output.size() * output[0].size() * 4 << std::endl;
}

// Main function demonstrating the 4-layer CNN
int main() {
    // Example input matrix (8x8)
    Matrix input;
    input.assign(image_size, std::vector<int16_t>(image_size, 1));

    // Example kernel (3x3)
    Matrix kernel1 = {
        {1, 1, -1},
        {1, 2, -1},
        {1, 3, -1}
    };
    Matrix kernel2 = {
        {-1, 1, 4},
        {1, 2, -1},
        {-2, 3, -1}
    };
    Matrix kernel3 = {
        {0, 1, -1},
        {1, 2, 1},
        {-1, 0, 1}
    };
    Matrix kernel4 = {
        {1, 0, 1},
        {-1, 1, 1},
        {1, 0, -1}
    };

    Matrix output;

    CNN4L(input, kernel1, kernel2, kernel3, kernel4, output);

    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            std::cout << output[i][j] << '\t';
        }
        std::cout << '\n';
    }

    std::cout << "finish" << std::endl;

    return 0;
}
