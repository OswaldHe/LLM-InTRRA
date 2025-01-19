#include <vector>
#include <cmath>
#include <iostream>
#include <string>
#include <ctime>
#include <cmath>
#include <tapa.h>
#include <gflags/gflags.h>
#include <ap_int.h>

#define VEC_LEN 32
constexpr int B = 32;  // Batch size
constexpr int ID = 4096; // Input dimension
constexpr int HD = 11008; // Hidden dimension

using type_t = ap_int<16>;
using vec_t = tapa::vec_t<type_t, VEC_LEN>;

void gating_net(
    tapa::mmap<vec_t> input,
    tapa::mmap<vec_t> W_up,
    tapa::mmap<vec_t> W_gate,
    tapa::mmap<vec_t> W_down,
    tapa::mmap<vec_t> combined,
    tapa::mmap<vec_t> output,
    tapa::mmap<int> cycle_count
);

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

DEFINE_string(bitstream, "", "path to bitstream file");

int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    srand((unsigned)time(nullptr));

    // Initialize input matrices
    LOG(INFO) << "Initializing input matrices...";
    aligned_vector<type_t> input(B * ID);
    for (int i = 0; i < B; ++i) {
        for (int j = 0; j < ID; ++j) {
            input[i * ID + j] = type_t(1);
        }
    }
    LOG(INFO) << "Input matrices initialized with all 1s";

    // Initialize weight matrices
    LOG(INFO) << "Initializing weight matrices...";
    aligned_vector<type_t> W_up(ID * HD);
    aligned_vector<type_t> W_gate(ID * HD);
    aligned_vector<type_t> W_down(HD * ID);  // don't need to transpose this
    for (int i = 0; i < ID; ++i) {
        for (int j = 0; j < HD; ++j) {
            W_up[j * ID + i] = type_t(1);   // Note: Transposed storage
            W_gate[j * ID + i] = type_t(1);  // Note: Transposed storage
        }
    }
    for (int i = 0; i < HD; ++i) {
        for (int j = 0; j < ID; ++j) {
            W_down[i * ID + j] = type_t(1);
        }
    }
    LOG(INFO) << "Weight matrices initialized with all 1s";

    // Output and cycle count
    aligned_vector<type_t> output(B * ID);
    aligned_vector<int> cycle_count(1);
    aligned_vector<type_t> combined(B * HD);

    // Invoke kernel
    int64_t kernel_time_ns = tapa::invoke(gating_net, FLAGS_bitstream,
        tapa::read_only_mmap<type_t>(input).reinterpret<vec_t>(),
        tapa::read_only_mmap<type_t>(W_up).reinterpret<vec_t>(),
        tapa::read_only_mmap<type_t>(W_gate).reinterpret<vec_t>(),
        tapa::read_only_mmap<type_t>(W_down).reinterpret<vec_t>(),
        tapa::write_only_mmap<type_t>(combined).reinterpret<vec_t>(),
        tapa::write_only_mmap<type_t>(output).reinterpret<vec_t>(),
        tapa::write_only_mmap<int>(cycle_count));

    std::cout << "Cycle count: " << cycle_count[0] << std::endl;
    std::cout << "Latency: " << kernel_time_ns * 1e-9 << " s" << std::endl;

    // for (int i = 0; i < B; i++) {
    //     for (int j = 0; j < ID; j++) {
    //         std::cout << output[i * ID + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    return 0;
}