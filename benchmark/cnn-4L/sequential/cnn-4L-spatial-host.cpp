#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <ctime>
#include <cmath>
#include <tapa.h>
#include <gflags/gflags.h>
#include <ap_int.h>

constexpr int image_shape = 224;
constexpr int image_size = image_shape*image_shape;

constexpr int kernel_shape = 3;
constexpr int kernel_size = kernel_shape*kernel_shape;

constexpr int layer1_output_shape = image_shape*2;
constexpr int layer1_output_size = layer1_output_shape*layer1_output_shape;

constexpr int layer2_output_shape = layer1_output_shape*2;
constexpr int layer2_output_size = layer2_output_shape*layer2_output_shape;

constexpr int layer3_output_shape = layer2_output_shape/2;
constexpr int layer3_output_size = layer3_output_shape*layer3_output_shape;

constexpr int layer4_output_shape = layer3_output_shape/2;
constexpr int layer4_output_size = layer4_output_shape*layer4_output_shape;

using int16_v16 = tapa::vec_t<ap_int<16>, 16>;

void CNN4L(
    tapa::mmap<int16_v16> input,
    tapa::mmap<int16_v16> kernel1,
    tapa::mmap<int16_v16> kernel2,
    tapa::mmap<int16_v16> kernel3,
    tapa::mmap<int16_v16> kernel4,
    tapa::mmap<int16_v16> offchip1,
    tapa::mmap<int16_v16> offchip2,
    tapa::mmap<int16_v16> offchip3,
    tapa::mmap<int16_v16> data_out,
    tapa::mmap<int> cycle_count
);

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

DEFINE_string(bitstream, "", "path to bitstream file");

int main(int argc, char *argv[]){
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    srand((unsigned)time(nullptr));

    // Example input matrix (8x8)
    aligned_vector<ap_int<16>> input(image_size);
    for (int i = 0; i < image_shape; ++i) {
        for (int j = 0; j < image_shape; ++j) {
            input[i*image_shape + j] = ap_int<16>(1);
        }
    }

    // Example kernel (3x3)
    aligned_vector<ap_int<16>> kernel1 = {
        1, 1, -1,
        1, 2, -1,
        1, 3, -1,
        0, 0, 0, 0, 0, 0, 0
    };
    aligned_vector<ap_int<16>> kernel2 = {
        -1, 1, 4,
        1, 2, -1,
        -2, 3, -1,
        0, 0, 0, 0, 0, 0, 0
    };
    aligned_vector<ap_int<16>> kernel3 = {
        0, 1, -1,
        1, 2, 1,
        -1, 0, 1,
        0, 0, 0, 0, 0, 0, 0
    };
    aligned_vector<ap_int<16>> kernel4 = {
        1, 0, 1,
        -1, 1, 1,
        1, 0, -1,
        0, 0, 0, 0, 0, 0, 0
    };

    // aligned_vector<ap_int<16>> output(image_size);
    const int s = layer4_output_shape;
    aligned_vector<ap_int<16>> output(s*s);
    aligned_vector<int> cycle_count(1);

    aligned_vector<ap_int<16>> offchip1(layer2_output_size);
    aligned_vector<ap_int<16>> offchip2(layer2_output_size);
    aligned_vector<ap_int<16>> offchip3(layer2_output_size);

    int64_t kernel_time_ns = tapa::invoke(CNN4L, FLAGS_bitstream,
        tapa::read_only_mmap<ap_int<16>>(input).reinterpret<int16_v16>(),
        tapa::read_only_mmap<ap_int<16>>(kernel1).reinterpret<int16_v16>(),
        tapa::read_only_mmap<ap_int<16>>(kernel2).reinterpret<int16_v16>(),
        tapa::read_only_mmap<ap_int<16>>(kernel3).reinterpret<int16_v16>(),
        tapa::read_only_mmap<ap_int<16>>(kernel4).reinterpret<int16_v16>(),
        tapa::read_write_mmap<ap_int<16>>(offchip1).reinterpret<int16_v16>(),
        tapa::read_write_mmap<ap_int<16>>(offchip2).reinterpret<int16_v16>(),
        tapa::read_write_mmap<ap_int<16>>(offchip3).reinterpret<int16_v16>(),
        tapa::write_only_mmap<ap_int<16>>(output).reinterpret<int16_v16>(),
        tapa::write_only_mmap<int>(cycle_count)
    );

    std::cout << "Cycle count: " << cycle_count[0] << std::endl;
    std::cout << "Kernel time (ns): " << kernel_time_ns << std::endl;
    std::cout << "Kernel time (us): " << float(kernel_time_ns)/1000.0 << std::endl;
    std::cout << "Kernel time (ms): " << float(kernel_time_ns)/1000000.0 << std::endl;

    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            std::cout << output[i*s + j] << '\t';
        }
        std::cout << '\n';
    }

    return 0;
}
