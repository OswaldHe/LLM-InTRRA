#include <tapa.h>
#include <ap_int.h>

constexpr int image_shape = 224;
constexpr int image_shape_div_16 = image_shape / 16;
constexpr int image_shape_div_32 = image_shape / 32;
constexpr int image_size = image_shape*image_shape;

constexpr int kernel_shape = 3;
constexpr int kernel_shape_mul_2 = kernel_shape * 2;
constexpr int kernel_shape_mul_4 = kernel_shape * 4;
constexpr int kernel_size = kernel_shape*kernel_shape;

constexpr int layer1_output_shape = image_shape*2;
constexpr int layer1_output_shape_div_2 = layer1_output_shape / 2;
constexpr int layer1_output_shape_div_16 = layer1_output_shape / 16;
constexpr int layer1_output_shape_div_32 = layer1_output_shape / 32;
constexpr int layer1_output_size = layer1_output_shape*layer1_output_shape;

constexpr int layer2_output_shape = layer1_output_shape*2;
constexpr int layer2_output_shape_div_2 = layer2_output_shape / 2;
constexpr int layer2_output_shape_div_16 = layer2_output_shape / 16;
constexpr int layer2_output_size = layer2_output_shape*layer2_output_shape;

constexpr int layer3_output_shape = layer2_output_shape/2;
constexpr int layer3_output_shape_div_2 = layer3_output_shape/2;
constexpr int layer3_output_shape_div_16 = layer3_output_shape/16;
constexpr int layer3_output_size = layer3_output_shape*layer3_output_shape;

constexpr int layer4_output_shape = layer3_output_shape/2;
constexpr int layer4_output_shape_div_16 = layer4_output_shape/16;
constexpr int layer4_output_size = layer4_output_shape*layer4_output_shape;

using int16_v16 = tapa::vec_t<ap_int<16>, 16>;

void measure_cycle(tapa::istreams<bool, 1>& fifo_fin, tapa::mmap<int> cycle_count){
    measure_cycle_loop: for (int cycle = 0;;cycle++){
        bool flag_cont = false;
        for(int i = 0; i < 1; i++){
            flag_cont |= fifo_fin[i].empty();
        }
        if(!flag_cont){
            measure_cycle_loop_count: for (int i = 0; i < 1; i++){
                fifo_fin[i].read(nullptr);
            }
            cycle_count[0] = cycle;
            break;
        }
    }
}

void top(
    tapa::async_mmap<int16_v16>& X,
    tapa::async_mmap<int16_v16>& kernel1_map,
    tapa::async_mmap<int16_v16>& kernel2_map,
    tapa::async_mmap<int16_v16>& kernel3_map,
    tapa::async_mmap<int16_v16>& kernel4_map,
    tapa::async_mmap<int16_v16>& offchip1,
    tapa::async_mmap<int16_v16>& offchip2,
    tapa::async_mmap<int16_v16>& offchip3,
    tapa::async_mmap<int16_v16>& data_out,
    tapa::ostream<bool>& fifo_fin
) {
    ap_int<16> tmp1[kernel_shape][image_shape];
    ap_int<16> tmp2[kernel_shape_mul_2][layer1_output_shape];
    ap_int<16> tmp3[kernel_shape_mul_4][layer2_output_shape];
    ap_int<16> tmp4[kernel_shape_mul_2][layer3_output_shape];
    #pragma HLS array_partition variable=tmp1 complete dim=1
    #pragma HLS array_partition variable=tmp2 complete dim=1
    #pragma HLS array_partition variable=tmp3 complete dim=1
    #pragma HLS array_partition variable=tmp4 complete dim=1

    #pragma HLS array_partition variable=tmp1 cyclic dim=2 factor=32
    #pragma HLS array_partition variable=tmp2 cyclic dim=2 factor=32
    #pragma HLS array_partition variable=tmp3 cyclic dim=2 factor=32
    #pragma HLS array_partition variable=tmp4 cyclic dim=2 factor=32

    #pragma HLS bind_storage variable=tmp1 type=RAM_2P impl=uram

    ap_int<16> kernel1[16];
    ap_int<16> kernel2[16];
    ap_int<16> kernel3[16];
    ap_int<16> kernel4[16];
    #pragma HLS array_partition variable=kernel1 complete
    #pragma HLS array_partition variable=kernel2 complete
    #pragma HLS array_partition variable=kernel3 complete
    #pragma HLS array_partition variable=kernel4 complete
    int16_v16 tmp_kernel;

    // Read kernel1
    kernel1_map.read_addr.write(0);
    for (;kernel1_map.read_data.try_read(tmp_kernel) == false;) { }
    for (int i = 0; i < 16; i++) {
        #pragma HLS unroll
        kernel1[i] = tmp_kernel[i];
    }
    // Read kernel2
    kernel2_map.read_addr.write(0);
    for (;kernel2_map.read_data.try_read(tmp_kernel) == false;) { }
    for (int i = 0; i < 16; i++) {
        #pragma HLS unroll
        kernel2[i] = tmp_kernel[i];
    }
    // Read kernel3
    kernel3_map.read_addr.write(0);
    for (;kernel3_map.read_data.try_read(tmp_kernel) == false;) { }
    for (int i = 0; i < 16; i++) {
        #pragma HLS unroll
        kernel3[i] = tmp_kernel[i];
    }
    // Read kernel4
    kernel4_map.read_addr.write(0);
    for (;kernel4_map.read_data.try_read(tmp_kernel) == false;) { }
    for (int i = 0; i < 16; i++) {
        #pragma HLS unroll
        kernel4[i] = tmp_kernel[i];
    }

    // Layer 1
    int row_cycle_index = 1;
    conv1_row_loop: for (int row = -1; row < image_shape; row++) {
        // Fetch row
        if (row < image_shape - 1) {
            conv1_fetch_row: for (int i_req = 0, i_resp = 0; i_resp < image_shape_div_16;) {
                #pragma HLS pipeline II=1
                if (i_req < image_shape_div_16 && X.read_addr.try_write((row+1)*image_shape_div_16 + i_req)) {
                    ++i_req;
                }
                if (!X.read_data.empty()) {
                    int16_v16 tmp = X.read_data.read(nullptr);
                    for (int j = 0; j < 16; j++) {
                        #pragma HLS unroll
                        tmp1[row_cycle_index][i_resp*16 + j] = tmp[j];
                    }
                    ++i_resp;
                }
            }
        }
        row_cycle_index = (row_cycle_index + 1) % kernel_shape;

        // Compute across row
        if (row >= 0) {
            conv1_compute_loop: for (int col_block = 0; col_block < image_shape_div_16; col_block++) {
                #pragma HLS pipeline II=1
                int16_v16 pkt;
                conv1_compute_unpack: for (int cc = 0; cc < 16; cc++) {
                    #pragma HLS unroll
                    int col = col_block*16 + cc;
                    ap_int<16> sum = 0;
                    for (int kr = 0; kr < kernel_shape; kr++) {
                        for (int kc = 0; kc < kernel_shape; kc++) {
                            if (!((col == 0 && kc == 0) || (col == (image_shape - 1) && kc == kernel_shape - 1) ||
                                (row == 0 && kr == 0) || (row == (image_shape - 1) && kr == kernel_shape - 1))
                            ) {
                                sum += kernel1[kr*kernel_shape + kc] * tmp1[(row_cycle_index + kr)%kernel_shape][col + kc - 1];
                            }
                        }
                    }
                    // ReLU
                    pkt[cc] = sum > ap_int<16>(0) ? sum : ap_int<16>(0);
                }

                // issue write request
                int addr = row*image_shape_div_16 + col_block;
                for (;offchip1.write_addr.full() || offchip1.write_data.full();) { }
                offchip1.write_addr.write(addr);
                offchip1.write_data.write(pkt);
                for (;offchip1.write_resp.empty();) { }
                offchip1.write_resp.read(nullptr);
            }
        }
    }

    // Layer 2
    row_cycle_index   = 1;
    int row_cycle_index_2 = 2;
    conv2_row_loop: for (int row = -1; row < layer1_output_shape_div_2; row++) {
        // Fetch row
        if (row < layer1_output_shape_div_2 - 1) {
            conv2_fetch_row: for (int i_req = 0, i_resp = 0; i_resp < image_shape_div_16;) {
                #pragma HLS DEPENDENCE variable=tmp2 inter WAW false
                #pragma HLS DEPENDENCE variable=tmp2 intra WAW false
                #pragma HLS pipeline II=1
                if (i_req < image_shape_div_16 && offchip1.read_addr.try_write((row+1)*image_shape_div_16 + i_req)) {
                    ++i_req;
                }
                if (!offchip1.read_data.empty()) {
                    int16_v16 tmp = offchip1.read_data.read(nullptr);
                    conv2_unpack_row: for (int j = 0; j < 16; j++) {
                        #pragma HLS unroll
                        tmp2[row_cycle_index  ][i_resp*32 + 2*j] = tmp[j];
                        tmp2[row_cycle_index  ][i_resp*32 + 2*j + 1] = tmp[j];
                        tmp2[row_cycle_index_2][i_resp*32 + 2*j] = tmp[j];
                        tmp2[row_cycle_index_2][i_resp*32 + 2*j + 1] = tmp[j];
                    }
                    ++i_resp;
                }
            }
        }
        row_cycle_index   = (row_cycle_index + 2)   % (kernel_shape_mul_2);
        row_cycle_index_2 = (row_cycle_index_2 + 2) % (kernel_shape_mul_2);

        // Compute across row
        if (row >= 0) {
            conv2_compute_two_rows: for (int rr = 0; rr < 2; rr++) {
                conv2_compute_loop: for (int col_block = 0; col_block < layer1_output_shape_div_16; col_block++) {
                    #pragma HLS pipeline II=1
                    int16_v16 pkt;
                    conv2_compute_unpack: for (int cc = 0; cc < 16; cc++) {
                        #pragma HLS unroll
                        int col = col_block*16 + cc;
                        ap_int<16> sum = 0;
                        for (int kr = 0; kr < kernel_shape; kr++) {
                            const int r = (row_cycle_index + 1 + rr + kr)%(kernel_shape_mul_2);

                            for (int kc = 0; kc < kernel_shape; kc++) {
                                if (!((col == 0 && kc == 0) || (col == (layer1_output_shape - 1) && kc == kernel_shape - 1) ||
                                    (row == 0 && rr == 0 && kr == 0) || (row == (layer1_output_shape_div_2 - 1) && rr == 1 && kr == kernel_shape - 1))
                                ) {
                                    sum += kernel2[kr*kernel_shape + kc] * tmp2[r][col + kc - 1];
                                }
                            }
                        }
                        pkt[cc] = sum > ap_int<16>(0) ? sum : ap_int<16>(0);
                    }

                    // issue write request
                    int addr = (2*row+rr)*layer1_output_shape_div_16 + col_block;
                    for (;offchip2.write_addr.full() || offchip2.write_data.full();) { }
                    offchip2.write_addr.write(addr);
                    offchip2.write_data.write(pkt);
                    for (;offchip2.write_resp.empty();) { }
                    offchip2.write_resp.read(nullptr);
                }
            }
        }
    }

    // Layer 3
    row_cycle_index   = 1;
    row_cycle_index_2 = 2;
    conv3_row_loop: for (int row = -1; row < layer2_output_shape_div_2; row++) {
        // Fetch row
        if (row < layer2_output_shape_div_2 - 1) {
            conv3_fetch_row: for (int i_req = 0, i_resp = 0; i_resp < layer1_output_shape_div_16;) {
                #pragma HLS DEPENDENCE variable=tmp3 inter WAW false
                #pragma HLS DEPENDENCE variable=tmp3 intra WAW false
                #pragma HLS pipeline II=1
                if (i_req < layer1_output_shape_div_16 && offchip2.read_addr.try_write((row+1)*layer1_output_shape_div_16 + i_req)) {
                    ++i_req;
                }
                if (!offchip2.read_data.empty()) {
                    int16_v16 tmp = offchip2.read_data.read(nullptr);
                    conv3_unpack_row: for (int j = 0; j < 16; j++) {
                        #pragma HLS unroll
                        tmp3[row_cycle_index  ][i_resp*32 + 2*j] = tmp[j];
                        tmp3[row_cycle_index  ][i_resp*32 + 2*j + 1] = tmp[j];
                        tmp3[row_cycle_index_2][i_resp*32 + 2*j] = tmp[j];
                        tmp3[row_cycle_index_2][i_resp*32 + 2*j + 1] = tmp[j];
                    }
                    ++i_resp;
                }
            }
        }
        row_cycle_index   = (row_cycle_index + 2)   % (kernel_shape_mul_4);
        row_cycle_index_2 = (row_cycle_index_2 + 2) % (kernel_shape_mul_4);

        // Compute across row
        if (row >= 0) {
            int16_v16 pkt_maxpool;
            for (int i = 0; i < 16; i++) {
                #pragma HLS unroll
                pkt_maxpool[i] = ap_int<16>(0);
            }
            conv3_compute_loop: for (int col_block = 0; col_block < layer2_output_shape_div_16; col_block++) {
                conv3_compute_two_rows: for (int rr = 0; rr < 2; rr++) {
                    #pragma HLS pipeline II=1
                    conv3_compute_unpack: for (int cc = 0; cc < 16; cc++) {
                        #pragma HLS unroll
                        int col = col_block*16 + cc;
                        ap_int<16> sum = 0;
                        for (int kr = 0; kr < kernel_shape; kr++) {
                            const int r = (row_cycle_index + 1 + rr + kr)%(kernel_shape_mul_4);

                            for (int kc = 0; kc < kernel_shape; kc++) {
                                if (!((col == 0 && kc == 0) || (col == (layer2_output_shape - 1) && kc == kernel_shape - 1) ||
                                    (row == 0 && rr == 0 && kr == 0) || (row == (layer2_output_shape_div_2 - 1) && rr == 1 && kr == kernel_shape - 1))
                                ) {
                                    sum += kernel3[kr*kernel_shape + kc] * tmp3[r][col + kc - 1];
                                }
                            }
                        }
                        ap_int<16> old = pkt_maxpool[(col_block%2)*8 + cc/2];
                        pkt_maxpool[(col_block%2)*8 + cc/2] = sum > old ? sum : old;
                    }
                }
                if (col_block % 2 == 1) {
                    // issue write request
                    int addr = row*layer3_output_shape_div_16 + col_block/2;
                    for (;offchip3.write_addr.full() || offchip3.write_data.full();) { }
                    offchip3.write_addr.write(addr);
                    offchip3.write_data.write(pkt_maxpool);
                    for (;offchip3.write_resp.empty();) { }
                    offchip3.write_resp.read(nullptr);

                    for (int i = 0; i < 16; i++) {
                        #pragma HLS unroll
                        pkt_maxpool[i] = ap_int<16>(0);
                    }
                }
            }
        }
    }

    // Layer 4
    row_cycle_index   = 1;
    conv4_row_loop: for (int row = -1; row < layer3_output_shape_div_2; row++) {
        // Fetch row
        if (row < layer3_output_shape_div_2 - 1) {
            for (int ii = 0; ii < 2; ii++) {
                int r = 2*(row+1) + ii;
                conv4_fetch_row: for (int i_req = 0, i_resp = 0; i_resp < layer3_output_shape_div_16;) {
                    #pragma HLS DEPENDENCE variable=tmp3 inter WAW false
                    #pragma HLS DEPENDENCE variable=tmp3 intra WAW false
                    #pragma HLS pipeline II=1
                    if (i_req < layer3_output_shape_div_16 && offchip3.read_addr.try_write(r*layer3_output_shape_div_16 + i_req)) {
                        ++i_req;
                    }
                    if (!offchip3.read_data.empty()) {
                        int16_v16 tmp = offchip3.read_data.read(nullptr);
                        conv4_unpack_row: for (int j = 0; j < 16; j++) {
                            #pragma HLS unroll
                            tmp4[(row_cycle_index+ii)%kernel_shape_mul_2][i_resp*16 + j] = tmp[j];
                        }
                        ++i_resp;
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
            conv4_compute_loop: for (int col_block = 0; col_block < layer3_output_shape_div_16; col_block++) {
                conv4_compute_two_rows: for (int rr = 0; rr < 2; rr++) {
                    #pragma HLS pipeline II=1
                    conv4_compute_unpack: for (int cc = 0; cc < 16; cc++) {
                        #pragma HLS unroll
                        int col = col_block*16 + cc;
                        ap_int<16> sum = 0;
                        for (int kr = 0; kr < kernel_shape; kr++) {
                            const int r = (row_cycle_index + 1 + rr + kr)%(kernel_shape_mul_2);

                            for (int kc = 0; kc < kernel_shape; kc++) {
                                if (!((col == 0 && kc == 0) || (col == (layer3_output_shape - 1) && kc == kernel_shape - 1) ||
                                    (row == 0 && rr == 0 && kr == 0) || (row == (layer3_output_shape_div_2 - 1) && rr == 1 && kr == kernel_shape - 1))
                                ) {
                                    sum += kernel4[kr*kernel_shape + kc] * tmp4[r][col + kc - 1];
                                }
                            }
                        }
                        ap_int<16> old = pkt_maxpool[(col_block%2)*8 + cc/2];
                        pkt_maxpool[(col_block%2)*8 + cc/2] = sum > old ? sum : old;
                    }
                }
                if (col_block % 2 == 1) {
                    // issue write request
                    int addr = row*layer4_output_shape_div_16 + col_block/2;
                    for (;data_out.write_addr.full() || data_out.write_data.full();) { }
                    data_out.write_addr.write(addr);
                    data_out.write_data.write(pkt_maxpool);
                    for (;data_out.write_resp.empty();) { }
                    data_out.write_resp.read(nullptr);

                    for (int i = 0; i < 16; i++) {
                        #pragma HLS unroll
                        pkt_maxpool[i] = ap_int<16>(0);
                    }
                }
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
    tapa::mmap<int16_v16> offchip1,
    tapa::mmap<int16_v16> offchip2,
    tapa::mmap<int16_v16> offchip3,
    tapa::mmap<int16_v16> data_out,
    tapa::mmap<int> cycle_count
) {
    tapa::streams<bool, 1> fifo_fin("fifo_fin");

    tapa::task()
        .invoke<tapa::join>(
            top,
            X,
            kernel1,
            kernel2,
            kernel3,
            kernel4,
            offchip1,
            offchip2,
            offchip3,
            data_out,
            fifo_fin
        )
        .invoke<tapa::join>(
            measure_cycle,
            fifo_fin,
            cycle_count
        )
    ;
}
