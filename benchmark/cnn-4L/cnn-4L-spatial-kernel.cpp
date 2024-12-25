#include <tapa.h>
#include <ap_int.h>

#define MEASURE_CYCLE_COUNT 1

constexpr int image_shape = 224;
constexpr int image_size = image_shape*image_shape;

constexpr int kernel_shape = 3;
constexpr int kernel_shape_mul_2 = kernel_shape*2;
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

template <typename data_t>
inline void bh(tapa::istream<data_t> & fifo_in) {
#pragma HLS inline
    bh: for (;;) {
#pragma HLS pipeline II=1 style=stp
        data_t tmp; fifo_in.try_read(tmp);
    }
}

void black_hole_int16_v16(tapa::istream<int16_v16>& fifo_in) {
    bh(fifo_in);
}

void read_input(
    const int input_size,
    tapa::async_mmap<int16_v16>& vec,
    tapa::ostream<int16_v16>& fifo_out
){
    const int bound = input_size >> 4;
    read_input_loop: for(int i_req = 0, i_resp = 0; i_resp < bound;){
        #pragma HLS pipeline II=1 style=stp
        if((i_req < bound) & !vec.read_addr.full()){
            vec.read_addr.write(i_req);
            i_req++;
        }
        int16_v16 tmp_o;
        bool success = vec.read_data.try_read(tmp_o);
        if(success){
            fifo_out.write(tmp_o);
            i_resp++;
        }
    }
}

void measure_cycle(tapa::istreams<bool, MEASURE_CYCLE_COUNT>& fifo_fin, tapa::mmap<int> cycle_count){
    measure_cycle_loop: for (int cycle = 0;;cycle++){
        bool flag_cont = false;
        for(int i = 0; i < MEASURE_CYCLE_COUNT; i++){
            flag_cont |= fifo_fin[i].empty();
        }
        if(!flag_cont){
            measure_cycle_loop_count: for (int i = 0; i < MEASURE_CYCLE_COUNT; i++){
                fifo_fin[i].read(nullptr);
            }
            cycle_count[0] = cycle;
            break;
        }
    }
}

void conv_layer1(
    tapa::istream<int16_v16>& fifo_kernel,
    tapa::istream<int16_v16>& fifo_input,
    tapa::ostream<int16_v16>& fifo_output
) {
    constexpr int input_shape = image_shape;
    constexpr int input_shape_div_16 = input_shape >> 4;

    ap_int<16> X[kernel_shape][input_shape];
    ap_int<16> kernel[16];
    #pragma HLS array_partition variable=X cyclic dim=2 factor=32
    #pragma HLS array_partition variable=kernel complete

    conv1_read_kernel: for (int i = 0; i < 1;) {
        #pragma HLS pipeline II=1 style=stp
        if(!fifo_kernel.empty()){
            int16_v16 tmp; fifo_kernel.try_read(tmp);
            conv1_read_kernel_unpack: for(int j = 0; j < 16; j++){
                #pragma HLS unroll
                kernel[i*16+j] = tmp[j];
            }
            i++;
        }
    }

    int row_cycle_index = 1;
    conv1_row_loop: for (int row = -1; row < input_shape; row++) {
        // Fetch row
        if (row < input_shape - 1) {
            conv1_fetch_row: for (int i = 0; i < input_shape_div_16;) {
                #pragma HLS pipeline II=1 style=stp
                if (!fifo_input.empty()) {
                    int16_v16 tmp; fifo_input.try_read(tmp);
                    conv1_unpack_row: for (int j = 0; j < 16; j++) {
                        #pragma HLS unroll
                        X[row_cycle_index][i*16 + j] = tmp[j];
                    }
                    i++;
                }
            }
        }
        row_cycle_index = (row_cycle_index + 1) % kernel_shape;

        // Compute across row
        if (row >= 0) {
            conv1_compute_loop: for (int col_block = 0; col_block < input_shape_div_16; col_block++) {
                #pragma HLS pipeline II=1
                int16_v16 pkt;
                conv1_compute_unpack: for (int cc = 0; cc < 16; cc++) {
                    #pragma HLS unroll
                    int col = col_block*16 + cc;
                    ap_int<16> sum = 0;
                    for (int kr = 0; kr < kernel_shape; kr++) {
                        for (int kc = 0; kc < kernel_shape; kc++) {
                            if (!((col == 0 && kc == 0) || (col == (input_shape - 1) && kc == kernel_shape - 1) ||
                                (row == 0 && kr == 0) || (row == (input_shape - 1) && kr == kernel_shape - 1))
                            ) {
                                sum += kernel[kr*kernel_shape + kc] * X[(row_cycle_index + kr)%kernel_shape][col + kc - 1];
                            }
                        }
                    }
                    // ReLU
                    pkt[cc] = sum > ap_int<16>(0) ? sum : ap_int<16>(0);
                }
                fifo_output.write(pkt);
            }
        }
    }
}

void conv_layer2(
    tapa::istream<int16_v16>& fifo_kernel,
    tapa::istream<int16_v16>& fifo_input,
    tapa::ostream<int16_v16>& fifo_output
) {
    constexpr int input_shape = layer1_output_shape;
    constexpr int input_shape_div_2 = input_shape >> 1;
    constexpr int input_shape_div_16 = input_shape >> 4;
    constexpr int input_shape_div_32 = input_shape >> 5;

    ap_int<16> X[kernel_shape_mul_2][input_shape];
    ap_int<16> kernel[16];
    #pragma HLS array_partition variable=X cyclic dim=2 factor=32
    #pragma HLS array_partition variable=kernel complete
    conv2_read_kernel: for (int i = 0; i < 1;) {
        #pragma HLS pipeline II=1 style=stp

        if(!fifo_kernel.empty()){
            int16_v16 tmp; fifo_kernel.try_read(tmp);
            conv2_read_kernel_unpack: for(int j = 0; j < 16; j++){
                #pragma HLS unroll
                kernel[i*16+j] = tmp[j];
            }
            i++;
        }
    }

    int row_cycle_index   = 1;
    int row_cycle_index_2 = 2;
    conv2_row_loop: for (int row = -1; row < input_shape_div_2; row++) {
        // Fetch row
        if (row < input_shape_div_2 - 1) {
            conv2_fetch_row: for (int i = 0; i < input_shape_div_32;) {
                #pragma HLS pipeline II=1 style=stp
                if (!fifo_input.empty()) {
                    int16_v16 tmp; fifo_input.try_read(tmp);
                    conv2_unpack_row: for (int j = 0; j < 16; j++) {
                        #pragma HLS unroll
                        X[row_cycle_index  ][i*32 + 2*j] = tmp[j];
                        X[row_cycle_index  ][i*32 + 2*j + 1] = tmp[j];
                        X[row_cycle_index_2][i*32 + 2*j] = tmp[j];
                        X[row_cycle_index_2][i*32 + 2*j + 1] = tmp[j];
                    }
                    i++;
                }
            }
        }
        row_cycle_index   = (row_cycle_index + 2)   % (kernel_shape_mul_2);
        row_cycle_index_2 = (row_cycle_index_2 + 2) % (kernel_shape_mul_2);

        // Compute across row
        if (row >= 0) {
            conv2_compute_two_rows: for (int rr = 0; rr < 2; rr++) {
                conv2_compute_loop: for (int col_block = 0; col_block < input_shape_div_16; col_block++) {
                    #pragma HLS pipeline II=1
                    int16_v16 pkt;
                    conv2_compute_unpack: for (int cc = 0; cc < 16; cc++) {
                        #pragma HLS unroll
                        int col = col_block*16 + cc;
                        ap_int<16> sum = 0;
                        for (int kr = 0; kr < kernel_shape; kr++) {
                            const int r = (row_cycle_index + 1 + rr + kr)%(kernel_shape_mul_2);

                            for (int kc = 0; kc < kernel_shape; kc++) {
                                if (!((col == 0 && kc == 0) || (col == (input_shape - 1) && kc == kernel_shape - 1) ||
                                    (row == 0 && rr == 0 && kr == 0) || (row == (input_shape_div_2 - 1) && rr == 1 && kr == kernel_shape - 1))
                                ) {
                                    sum += kernel[kr*kernel_shape + kc] * X[r][col + kc - 1];
                                }
                            }
                        }
                        // ReLU
                        pkt[cc] = sum > ap_int<16>(0) ? sum : ap_int<16>(0);
                    }
                    fifo_output.write(pkt);
                }
            }
        }
    }
}

void conv_layer3(
    tapa::istream<int16_v16>& fifo_kernel,
    tapa::istream<int16_v16>& fifo_input,
    tapa::ostream<int16_v16>& fifo_output
) {
    constexpr int input_shape = layer2_output_shape;
    constexpr int input_shape_div_2 = input_shape >> 1;
    constexpr int input_shape_div_16 = input_shape >> 4;
    constexpr int input_shape_div_32 = input_shape >> 5;

    ap_int<16> X[2*kernel_shape][input_shape];
    ap_int<16> kernel[16];
    #pragma HLS array_partition variable=X cyclic dim=2 factor=32
    #pragma HLS array_partition variable=kernel complete

    conv3_read_kernel: for (int i = 0; i < 1;) {
        #pragma HLS pipeline II=1 style=stp
        if(!fifo_kernel.empty()){
            int16_v16 tmp; fifo_kernel.try_read(tmp);
            conv3_read_kernel_unpack: for(int j = 0; j < 16; j++){
                #pragma HLS unroll
                kernel[i*16+j] = tmp[j];
            }
            i++;
        }
    }

    int row_cycle_index = 1;
    int row_cycle_index_2 = 2;
    conv3_row_loop: for (int row = -1; row < input_shape_div_2; row++) {
        // Fetch row
        if (row < input_shape_div_2 - 1) {
            conv3_fetch_row: for (int i = 0; i < input_shape_div_32;) {
                #pragma HLS pipeline II=1 style=stp
                if (!fifo_input.empty()) {
                    int16_v16 tmp; fifo_input.try_read(tmp);
                    conv3_unpack_row: for (int j = 0; j < 16; j++) {
                        #pragma HLS unroll
                        X[row_cycle_index  ][i*32 + 2*j] = tmp[j];
                        X[row_cycle_index  ][i*32 + 2*j + 1] = tmp[j];
                        X[row_cycle_index_2][i*32 + 2*j] = tmp[j];
                        X[row_cycle_index_2][i*32 + 2*j + 1] = tmp[j];
                    }
                    i++;
                }
            }
        }
        row_cycle_index   = (row_cycle_index + 2)   % (kernel_shape_mul_2);
        row_cycle_index_2 = (row_cycle_index_2 + 2) % (kernel_shape_mul_2);

        // Compute across row
        if (row >= 0) {
            int16_v16 pkt_maxpool;
            for (int i = 0; i < 16; i++) {
                #pragma HLS unroll
                pkt_maxpool[i] = ap_int<16>(0);
            }
            conv3_compute_loop: for (int col_block = 0; col_block < input_shape_div_16; col_block++) {
                conv3_compute_two_rows: for (int rr = 0; rr < 2; rr++) {
                    #pragma HLS pipeline II=1
                    conv3_compute_unpack: for (int cc = 0; cc < 16; cc++) {
                        #pragma HLS unroll
                        int col = col_block*16 + cc;
                        ap_int<16> sum = 0;
                        for (int kr = 0; kr < kernel_shape; kr++) {
                            const int r = (row_cycle_index + 1 + rr + kr)%(kernel_shape_mul_2);

                            for (int kc = 0; kc < kernel_shape; kc++) {
                                if (!((col == 0 && kc == 0) || (col == (input_shape - 1) && kc == kernel_shape - 1) ||
                                    (row == 0 && rr == 0 && kr == 0) || (row == (input_shape_div_2 - 1) && rr == 1 && kr == kernel_shape - 1))
                                ) {
                                    sum += kernel[kr*kernel_shape + kc] * X[r][col + kc - 1];
                                }
                            }
                        }
                        ap_int<16> old = pkt_maxpool[(col_block%2)*8 + cc/2];
                        pkt_maxpool[(col_block%2)*8 + cc/2] = sum > old ? sum : old;
                    }
                }
                if (col_block % 2 == 1) {
                    // issue write request
                    fifo_output.write(pkt_maxpool);

                    for (int i = 0; i < 16; i++) {
                        #pragma HLS unroll
                        pkt_maxpool[i] = ap_int<16>(0);
                    }
                }
            }
        }
    }
}

void conv_layer4(
    tapa::istream<int16_v16>& fifo_kernel,
    tapa::istream<int16_v16>& fifo_input,
    tapa::ostream<int16_v16>& fifo_output
) {
    constexpr int input_shape = layer3_output_shape;
    constexpr int input_shape_div_2 = input_shape >> 1;
    constexpr int input_shape_div_16 = input_shape >> 4;

    ap_int<16> X[2*kernel_shape][input_shape];
    ap_int<16> kernel[16];
    #pragma HLS array_partition variable=X cyclic dim=2 factor=32
    #pragma HLS array_partition variable=kernel complete

    conv4_read_kernel: for (int i = 0; i < 1;) {
        #pragma HLS pipeline II=1 style=stp
        if(!fifo_kernel.empty()){
            int16_v16 tmp; fifo_kernel.try_read(tmp);
            conv4_read_kernel_unpack: for(int j = 0; j < 16; j++){
                #pragma HLS unroll
                kernel[i*16+j] = tmp[j];
            }
            i++;
        }
    }

    int row_cycle_index = 1;
    conv4_row_loop: for (int row = -1; row < input_shape_div_2; row++) {
        // Fetch row
        if (row < input_shape_div_2 - 1) {
            for (int ii = 0; ii < 2; ii++) {
                conv4_fetch_row: for (int i = 0; i < input_shape_div_16;) {
                    #pragma HLS pipeline II=1 style=stp
                    if (!fifo_input.empty()) {
                        int16_v16 tmp; fifo_input.try_read(tmp);
                        conv4_unpack_row: for (int j = 0; j < 16; j++) {
                            #pragma HLS unroll
                            X[(row_cycle_index+ii)%(kernel_shape_mul_2)][i*16 + j] = tmp[j];
                        }
                        i++;
                    }
                }
            }
        }
        row_cycle_index = (row_cycle_index + 2) % (kernel_shape_mul_2);

        // Compute across row
        if (row >= 0) {
            int16_v16 pkt_maxpool;
            for (int i = 0; i < 16; i++) {
                #pragma HLS unroll
                pkt_maxpool[i] = ap_int<16>(0);
            }
            conv4_compute_loop: for (int col_block = 0; col_block < input_shape_div_16; col_block++) {
                conv4_compute_two_rows: for (int rr = 0; rr < 2; rr++) {
                    #pragma HLS pipeline II=1
                    conv4_compute_unpack: for (int cc = 0; cc < 16; cc++) {
                        #pragma HLS unroll
                        int col = col_block*16 + cc;
                        ap_int<16> sum = 0;
                        for (int kr = 0; kr < kernel_shape; kr++) {
                            const int r = (row_cycle_index + 1 + rr + kr)%(kernel_shape_mul_2);

                            for (int kc = 0; kc < kernel_shape; kc++) {
                                if (!((col == 0 && kc == 0) || (col == (input_shape - 1) && kc == kernel_shape - 1) ||
                                    (row == 0 && rr == 0 && kr == 0) || (row == (input_shape_div_2 - 1) && rr == 1 && kr == kernel_shape - 1))
                                ) {
                                    sum += kernel[kr*kernel_shape + kc] * X[r][col + kc - 1];
                                }
                            }
                        }
                        ap_int<16> old = pkt_maxpool[(col_block%2)*8 + cc/2];
                        pkt_maxpool[(col_block%2)*8 + cc/2] = sum > old ? sum : old;
                    }
                }
                if (col_block % 2 == 1) {
                    // issue write request
                    fifo_output.write(pkt_maxpool);

                    for (int i = 0; i < 16; i++) {
                        #pragma HLS unroll
                        pkt_maxpool[i] = ap_int<16>(0);
                    }
                }
            }
        }
    }
}

void write_mtx(
    const int output_shape,
    tapa::async_mmap<int16_v16>& output_mtx,
    tapa::istream<int16_v16>& fifo_in,
    tapa::ostream<bool>& fifo_fin
){
    const int output_shape_div_16 = output_shape >> 4;
    write_mtx_row_loop: for (int row = 0; row < output_shape; row++) {
        int rr = row*output_shape_div_16;
        write_mtx_col_loop: for (int i_req = 0, i_resp = 0; i_resp < output_shape_div_16;) {
            #pragma HLS pipeline II=1 style=stp
            if((i_req < output_shape_div_16) & !fifo_in.empty() & !output_mtx.write_addr.full() & !output_mtx.write_data.full()){
                output_mtx.write_addr.try_write(rr + i_req);
                int16_v16 tmp; fifo_in.try_read(tmp);
                output_mtx.write_data.try_write(tmp);
                ++i_req;
            }
            bool success = false;
            auto resp = output_mtx.write_resp.read(success);
            if(success){
                i_resp += unsigned(resp)+1;
            }
        }
    }

    fifo_fin.write(true);
}

void CNN4L(
    tapa::mmap<int16_v16> X,
    tapa::mmap<int16_v16> kernel1,
    tapa::mmap<int16_v16> kernel2,
    tapa::mmap<int16_v16> kernel3,
    tapa::mmap<int16_v16> kernel4,
    tapa::mmap<int16_v16> data_out,
    tapa::mmap<int> cycle_count
) {
    tapa::stream<int16_v16, 8> fifo_X("fifo_X");
    tapa::stream<int16_v16> fifo_kernel1("fifo_kernel1");
    tapa::stream<int16_v16> fifo_kernel2("fifo_kernel2");
    tapa::stream<int16_v16> fifo_kernel3("fifo_kernel3");
    tapa::stream<int16_v16> fifo_kernel4("fifo_kernel4");

    tapa::stream<int16_v16> fifo_conv1("fifo_conv1");

    tapa::stream<int16_v16> fifo_conv2("fifo_conv2");

    tapa::stream<int16_v16> fifo_conv3("fifo_conv3");

    tapa::stream<int16_v16> fifo_conv4("fifo_conv4");

    tapa::streams<bool, MEASURE_CYCLE_COUNT> fifo_fin("fifo_fin");

    tapa::task()
        .invoke<tapa::join>(
            read_input,
            image_size,
            X,
            fifo_X
        )
        .invoke<tapa::join>(
            read_input,
            16,
            kernel1,
            fifo_kernel1
        )
        .invoke<tapa::join>(
            read_input,
            16,
            kernel2,
            fifo_kernel2
        )
        .invoke<tapa::join>(
            read_input,
            16,
            kernel3,
            fifo_kernel3
        )
        .invoke<tapa::join>(
            read_input,
            16,
            kernel4,
            fifo_kernel4
        )

        // Layer 1
        .invoke<tapa::join>(
            conv_layer1,
            fifo_kernel1,
            fifo_X,
            fifo_conv1
        )

        // Layer 2
        .invoke<tapa::join>(
            conv_layer2,
            fifo_kernel2,
            fifo_conv1,
            fifo_conv2
        )

        // Layer 3
        .invoke<tapa::join>(
            conv_layer3,
            fifo_kernel3,
            fifo_conv2,
            fifo_conv3
        )

        // Layer 4
        .invoke<tapa::join>(
            conv_layer4,
            fifo_kernel4,
            fifo_conv3,
            fifo_conv4
        )
        .invoke<tapa::join>(
            write_mtx,
            layer4_output_shape,
            data_out,
            fifo_conv4,
            fifo_fin
        )
        .invoke<tapa::join>(
            measure_cycle,
            fifo_fin,
            cycle_count
        )
    ;
}
