#include <iostream>
#include <cmath>
#include <numeric>
#include <string>
#include <tapa.h>
#include <ap_int.h>
#include <hls_math.h>

constexpr int image_shape = 224;
constexpr int image_size = image_shape * image_shape;
constexpr int kernel_shape = 3;
constexpr int kernel_size = kernel_shape * kernel_shape;

constexpr int kernel_shape_mul_2 = kernel_shape*2;

constexpr int layer1_output_shape = image_shape*2;
constexpr int layer1_output_size = layer1_output_shape*layer1_output_shape;

constexpr int layer2_output_shape = layer1_output_shape*2;
constexpr int layer2_output_size = layer2_output_shape*layer2_output_shape;

constexpr int layer3_output_shape = layer2_output_shape/2;
constexpr int layer3_output_size = layer3_output_shape*layer3_output_shape;

constexpr int layer4_output_shape = layer3_output_shape/2;
constexpr int layer4_output_size = layer4_output_shape*layer4_output_shape;

constexpr int output_size = 150 * layer3_output_shape;


using int16_v16 = tapa::vec_t<ap_int<16>, 16>;
using int16_v32 = tapa::vec_t<ap_int<16>, 32>;

void CNN4L(
    tapa::mmap<int16_v16> X,
    tapa::mmap<int16_v16> kernel1,
    tapa::mmap<int16_v16> kernel2,
    tapa::mmap<int16_v16> kernel3,
    tapa::mmaps<int16_v16, 3> data_out,
    tapa::mmap<int> cycle_count
);

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

DEFINE_string(bitstream, "", "path to bitstream file");

int main(int argc, char *argv[]){
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    const int L = argc > 1 ? atoll(argv[1]) : image_size;

    srand((unsigned)time(nullptr));

    // Example input and weight matrices
    
    aligned_vector<ap_int<16>> input1(image_size);
    aligned_vector<ap_int<16>> weight_cc0(32);
    aligned_vector<ap_int<16>> weight_cc1(32);
    aligned_vector<ap_int<16>> weight_cc2(32);
    aligned_vector<aligned_vector<ap_int<16>>> data_out(3, aligned_vector<ap_int<16>>(output_size));
    aligned_vector<int> cycle_count(1);

    int64_t kernel_time_ns = tapa::invoke(CNN4L, FLAGS_bitstream,
        tapa::read_only_mmap<ap_int<16>>(input1).reinterpret<int16_v16>(),
        tapa::read_only_mmap<ap_int<16>>(weight_cc0).reinterpret<int16_v16>(),
        tapa::read_only_mmap<ap_int<16>>(weight_cc1).reinterpret<int16_v16>(),
        tapa::read_only_mmap<ap_int<16>>(weight_cc1).reinterpret<int16_v16>(),
        tapa::write_only_mmaps<ap_int<16>, 3>(data_out).reinterpret<int16_v16>(),
        tapa::write_only_mmap<int>(cycle_count));

    std::cout << "Cycle count: " << cycle_count[0] << std::endl;
    std::cout << "Latency: " << kernel_time_ns * 1e-9 << " s" << std::endl;

    return 0;
}