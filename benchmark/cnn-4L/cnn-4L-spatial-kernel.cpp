#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <tapa.h>
#include <ap_int.h>

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

void read_W(
    const int w_size,
    tapa::async_mmap<int16_v16>& vec,
    tapa::ostream<int16_v16>& fifo_out
){
    for(int i_req = 0, i_resp = 0; i_resp < w_size;){
        #pragma HLS pipeline II=1 style=stp
        if((i_req < w_size) & !vec.read_addr.full()){
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

void read_X(
    tapa::async_mmap<int16_v16>& vec,
    tapa::ostream<int16_v16>& fifo_out
){
    for(int i_req = 0, i_resp = 0; i_resp < (image_size >> 4);){
        #pragma HLS pipeline II=1 style=stp
        if((i_req < (image_size >> 4)) & !vec.read_addr.full()){
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

void write_mtx(
    tapa::async_mmap<int16_v16>& output_mtx,
    tapa::istream<int16_v16>& fifo_in,
    tapa::ostream<bool>& fifo_fin
){

    for(int i_req = 0, i_resp = 0; i_resp < (output_size >> 4);){
        #pragma HLS pipeline II=1 style=stp
        if((i_req < (output_size >> 4)) & !fifo_in.empty() & !output_mtx.write_addr.full() & !output_mtx.write_data.full()){
            output_mtx.write_addr.try_write(i_req);
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

    fifo_fin.write(true);
} 

void CC0_Conv1_Conv4(
    tapa::istream<int16_v16>& fifo_input,
    tapa::istream<int16_v16>& fifo_kernel,
    tapa::ostream<int16_v16>& fifo_to_CC1,
    tapa::istream<int16_v16>& fifo_from_ctrl,
    tapa::ostream<int16_v16>& fifo_output
){

    ap_int<16> X[kernel_shape][layer3_output_shape];
    ap_int<16> kernel[16];
    #pragma HLS array_partition variable=X cyclic dim=2 factor=32
    #pragma HLS array_partition variable=kernel complete

    for(int st = 0; st < 2; st++){

        const int row_bound = (st == 0) ? image_shape : 150;
        const int col_bound = (st == 0) ? image_shape : layer3_output_shape;
        const int fetch_bound = (col_bound >> 4);

        int16_v16 tmp = fifo_kernel.read();
        for(int j = 0; j < 16; j++){
            #pragma HLS unroll
            kernel[j] = tmp[j];
        }

        if(st == 0){
            // initialize
            for(int i = 0; i < (image_shape >> 5); i++){
                #pragma HLS pipeline II=1
                for(int j = 0; j < 32; j++){
                    #pragma HLS unroll
                    X[0][i*32+j] = 0;
                }
            }
        }

        int row_cycle_index = 1;
        for (int row = -1; row < row_bound; row++) {
            // Fetch row
            if (row < row_bound - 1) {
                for (int i = 0; i < fetch_bound; i++) {
                    #pragma HLS pipeline II=1 style=stp
                    int16_v16 tmp;
                    if(st == 0){
                        tmp = fifo_input.read();
                    } else {
                        tmp = fifo_from_ctrl.read();
                    }
                    conv1_unpack_row: for (int j = 0; j < 16; j++) {
                        #pragma HLS unroll
                        X[row_cycle_index][i*16 + j] = tmp[j];
                    }
                }
            }
            row_cycle_index = (row_cycle_index + 1) % kernel_shape;

            // Compute across row
            if (row >= 0) {
                conv1_compute_loop: for (int col_block = 0; col_block < (col_bound >> 4); col_block++) {
                    #pragma HLS pipeline II=1
                    int16_v16 pkt;
                    conv1_compute_unpack: for (int cc = 0; cc < 16; cc++) {
                        #pragma HLS unroll
                        int col = col_block*16 + cc;
                        ap_int<16> sum = 0;
                        conv1_compute_kernel_row: for (int kr = 0; kr < kernel_shape; kr++) {
                            conv1_compute_kernel_col: for (int kc = 0; kc < kernel_shape; kc++) {
                                if (!((col == 0 && kc == 0) || (col == (col_bound - 1) && kc == kernel_shape - 1) ||
                                    (row == 0 && kr == 0) || (row == (row_bound - 1) && kr == kernel_shape - 1))
                                ) {
                                    sum += kernel[kr*kernel_shape + kc] * X[(row_cycle_index + kr)%kernel_shape][col + kc - 1];
                                }
                            }
                        }
                        pkt[cc] = sum;
                    }
                    if(st == 0){
                        fifo_to_CC1.write(pkt);
                    } else {
                        fifo_output.write(pkt);
                    }
                }
            }
        }
    }

}

void CC1_Conv2_Conv4(
    tapa::istreams<int16_v32, 2>& fifo_input,
    tapa::istream<int16_v16>& fifo_kernel,
    tapa::ostream<int16_v16>& fifo_to_CC2,
    tapa::istream<int16_v16>& fifo_from_ctrl,
    tapa::ostream<int16_v16>& fifo_output
){

    ap_int<16> X[kernel_shape_mul_2][layer1_output_shape];
    ap_int<16> kernel[16];
    #pragma HLS array_partition variable=X cyclic dim=2 factor=32
    #pragma HLS array_partition variable=X complete dim=1
    #pragma HLS array_partition variable=kernel complete

    for(int st = 0; st < 2; st++){

        const int row_bound = (st == 0) ? (layer1_output_shape / 2) : 150;
        const int col_bound = (st == 0) ? layer1_output_shape : layer3_output_shape;
        const int fetch_bound = (st == 0) ? (layer1_output_shape >> 5) : (layer3_output_shape >> 4);
        const int rr_bound = (st == 0) ? 2 : 1;

        int16_v16 tmp = fifo_kernel.read();
        for(int j = 0; j < 16; j++){
            #pragma HLS unroll
            kernel[j] = tmp[j];
        }

        if(st == 0){
            for(int i = 0; i < (col_bound >> 5); i++){
                #pragma HLS pipeline II=1
                for(int j = 0; j < 32; j++){
                    #pragma HLS unroll
                    X[0][i*32+j] = 0;
                }
            }
        }

        int row_cycle_index   = 1;
        int row_cycle_index_2 = 2;
        for (int row = -1; row < row_bound; row++) {
            // Fetch row
            if (row < row_bound - 1) {
                if(st == 0) {
                    for (int i = 0; i < fetch_bound;) {
                        #pragma HLS pipeline II=1 style=stp
                        #pragma HLS dependence variable=X false
                        if (!fifo_input[0].empty() && !fifo_input[1].empty()) {
                            int16_v32 tmp0; fifo_input[0].try_read(tmp0);
                            int16_v32 tmp1; fifo_input[1].try_read(tmp1);
                            conv2_unpack_row: for (int j = 0; j < 32; j++) {
                                #pragma HLS unroll
                                X[row_cycle_index  ][i*32 + j] = tmp0[j];
                                X[row_cycle_index_2][i*32 + j] = tmp1[j];
                            }
                            i++;
                        }
                    }
                } else {
                    conv1_fetch_row: for (int i = 0; i < fetch_bound;) {
                        #pragma HLS pipeline II=1 style=stp
                        if (!fifo_from_ctrl.empty()) {
                            int16_v16 tmp; fifo_from_ctrl.try_read(tmp);
                            conv1_unpack_row: for (int j = 0; j < 16; j++) {
                                #pragma HLS unroll
                                X[row_cycle_index][i*16 + j] = tmp[j];
                            }
                            i++;
                        }
                    }
                }
            }
            row_cycle_index   = (st == 0) ? ((row_cycle_index + 2) % kernel_shape_mul_2) : ((row_cycle_index + 1) % kernel_shape);
            row_cycle_index_2 = (row_cycle_index_2 + 2) % (kernel_shape_mul_2);

            // Compute across row
            if (row >= 0) {
                for (int rr = 0; rr < rr_bound; rr++) {
                    bool check_st = (rr == 1 || st == 1);
                    conv2_compute_loop: for (int col_block = 0; col_block < (col_bound >> 4); col_block++) {
                        #pragma HLS pipeline II=1
                        int16_v16 pkt;
                        conv2_compute_unpack: for (int cc = 0; cc < 16; cc++) {
                            #pragma HLS unroll
                            int col = col_block*16 + cc;
                            ap_int<16> sum = 0;
                            conv2_compute_kernel_row: for (int kr = 0; kr < kernel_shape; kr++) {
                                const int r = (st == 0) ? ((row_cycle_index + 1 + rr + kr)%kernel_shape_mul_2) : ((row_cycle_index + kr)%kernel_shape);

                                conv2_compute_kernel_col: for (int kc = 0; kc < kernel_shape; kc++) {
                                    if (!((col == 0 && kc == 0) || (col == (col_bound - 1) && kc == kernel_shape - 1) ||
                                        (row == 0 && rr == 0 && kr == 0) || (row == (row_bound - 1) && check_st && kr == kernel_shape - 1))
                                    ) {
                                        sum += kernel[kr*kernel_shape + kc] * X[r][col + kc - 1];
                                    }
                                }
                            }
                            pkt[cc] = sum;
                        }
                        if(st == 0){
                            fifo_to_CC2.write(pkt);
                        } else {
                            fifo_output.write(pkt);
                        }
                    }
                }
            }
        }
    }

}

void CC2_Conv3_Conv4(
    tapa::istreams<int16_v32, 2>& fifo_input,
    tapa::istream<int16_v16>& fifo_kernel,
    tapa::ostreams<int16_v16, 2>& fifo_to_ctrl,
    tapa::istream<int16_v16>& fifo_from_ctrl,
    tapa::ostream<int16_v16>& fifo_output
){

    ap_int<16> X[2*kernel_shape][layer2_output_shape];
    ap_int<16> kernel[16];
    #pragma HLS array_partition variable=X cyclic dim=2 factor=32
    #pragma HLS array_partition variable=X complete dim=1
    #pragma HLS array_partition variable=kernel complete


    for(int st = 0; st < 2; st++) {

        const int row_bound = (st == 0) ? (layer2_output_shape / 2) : 150;
        const int col_bound = (st == 0) ? layer2_output_shape : layer3_output_shape;
        const int fetch_bound = (st == 0) ? (layer2_output_shape >> 5) : (layer3_output_shape >> 4);
        const int rr_bound = (st == 0) ? 2 : 1;

        int16_v16 tmp = fifo_kernel.read();
        for(int j = 0; j < 16; j++){
            #pragma HLS unroll
            kernel[j] = tmp[j];
        }

        if(st == 0){
            for(int i = 0; i < (col_bound >> 5); i++){
                #pragma HLS pipeline II=1
                for(int j = 0; j < 32; j++){
                    #pragma HLS unroll
                    X[0][i*32+j] = 0;
                }
            }
        }

        int row_cycle_index = 1;
        conv3_row_loop: for (int row = -1; row < row_bound; row++) {
            // Fetch row
            if (row < row_bound - 1) {
                if(st == 0){
                    for (int i = 0; i < fetch_bound;) {
                        #pragma HLS pipeline II=1 style=stp
                        #pragma HLS dependence variable=X false
                        if (!fifo_input[0].empty() && !fifo_input[1].empty()) {
                            int16_v32 tmp0; fifo_input[0].try_read(tmp0);
                            int16_v32 tmp1; fifo_input[1].try_read(tmp1);
                            for (int j = 0; j < 32; j++) {
                                #pragma HLS unroll
                                X[row_cycle_index][i*32 + j] = tmp0[j];
                                X[(row_cycle_index+1)%(kernel_shape_mul_2)][i*32 + j] = tmp1[j];
                            }
                            i++;
                        }
                    }
                } else {
                    conv1_fetch_row: for (int i = 0; i < fetch_bound;) {
                        #pragma HLS pipeline II=1 style=stp
                        if (!fifo_from_ctrl.empty()) {
                            int16_v16 tmp; fifo_from_ctrl.try_read(tmp);
                            conv1_unpack_row: for (int j = 0; j < 16; j++) {
                                #pragma HLS unroll
                                X[row_cycle_index][i*16 + j] = tmp[j];
                            }
                            i++;
                        }
                    }
                }
            }
            row_cycle_index = (st == 0) ? ((row_cycle_index + 2) % kernel_shape_mul_2) : ((row_cycle_index + 1) % kernel_shape);

            // Compute across row
            if (row >= 0) {
                conv3_compute_loop: for (int col_block = 0; col_block < (col_bound >> 4); col_block++) {
                    for (int rr = 0; rr < rr_bound; rr++) {
                        #pragma HLS pipeline II=1

                        bool check_st = (rr == 1 || st == 1);

                        int16_v16 pkt;
                        for (int cc = 0; cc < 16; cc++) {
                            #pragma HLS unroll
                            int col = col_block*16 + cc;
                            ap_int<16> sum = 0;
                            for (int kr = 0; kr < kernel_shape; kr++) {
                                const int r = (st == 0) ? ((row_cycle_index + 1 + rr + kr)%kernel_shape_mul_2) : ((row_cycle_index + kr)%kernel_shape);
                                for (int kc = 0; kc < kernel_shape; kc++) {
                                    if (!((col == 0 && kc == 0) || (col == (col_bound - 1) && kc == kernel_shape - 1) ||
                                        (row == 0 && rr == 0 && kr == 0) || (row == (row_bound - 1) && check_st && kr == kernel_shape - 1))
                                    ) {
                                        sum += kernel[kr*kernel_shape + kc] * X[r][col + kc - 1];
                                    }
                                }
                            }
                            pkt[cc] = sum;
                        }
                        if(st == 0){
                            fifo_to_ctrl[rr].write(pkt);
                        } else {
                            fifo_output.write(pkt);
                        }
                    }
                }
            }
        }
    }
}

void central_mem(
    tapa::istream<int16_v16>& fifo_from_CC2,
    tapa::ostreams<int16_v16, 3>& fifo_to_CC
){
    ap_int<16> X[layer3_output_shape][layer3_output_shape];
    #pragma HLS array_partition variable=X cyclic factor=16 dim=2
    #pragma HLS array_partition variable=X cyclic factor=3 dim=1
    #pragma HLS bind_storage variable=X type=ram_2p impl=uram

    for(int i = 0; i < layer3_output_shape; i++){
        for(int j = 0; j < (layer3_output_shape >> 4);){
            if(!fifo_from_CC2.empty()){
                int16_v16 tmp; fifo_from_CC2.try_read(tmp);
                for(int k = 0; k < 16; k++){
                    X[i][j*16+k] = tmp[k];
                }
                j++;
            }
        }
    }

    // split into 3 batch
    const int start_idx_1 = 0;
    const int start_idx_2 = 149;
    const int start_idx_3 = 298;

    for(int i = 0; i < 150; i++){
        for(int j = 0; j < (layer3_output_shape >> 4); j++){
            #pragma HLS pipeline II=1

            int16_v16 pkt1;
            int16_v16 pkt2;
            int16_v16 pkt3;
            for(int k = 0; k < 16; k++){
                #pragma HLS unroll
                pkt1[k] = X[start_idx_1+i][j*16+k];
                pkt2[k] = X[start_idx_2+i][j*16+k];
                pkt3[k] = X[start_idx_3+i][j*16+k];
            }

            fifo_to_CC[0].write(pkt1);
            fifo_to_CC[1].write(pkt2);
            fifo_to_CC[2].write(pkt3);
        }
    }

}

void relu(
    tapa::istream<int16_v16>& fifo_input,
    tapa::ostream<int16_v16>& fifo_output
) {
    for(;;){
        if(!fifo_input.empty()){
            int16_v16 tmp; fifo_input.try_read(tmp);
            for(int i = 0; i < 16; i++){
                #pragma HLS unroll
                tmp[i] = (tmp[i] > 0) ? tmp[i] : ap_int<16>(0);
            }
            fifo_output.write(tmp);
        }
    }
}

void upsample(
    tapa::istream<int16_v16>& fifo_input,
    tapa::ostreams<int16_v32, 2>& fifo_output
) {
    upsample_loop: for (;;) {
        #pragma HLS pipeline II=1 style=stp
        if(!fifo_input.empty()){
            int16_v16 tmp; fifo_input.try_read(tmp);
            int16_v32 upsampled0;
            int16_v32 upsampled1;
            for (int jj = 0; jj < 16; jj++) {
                ap_int<16> val = tmp[jj];
                upsampled0[2*jj] = val;
                upsampled0[2*jj + 1] = val;
                upsampled1[2*jj] = val;
                upsampled1[2*jj + 1] = val;
            }
            fifo_output[0].write(upsampled0);
            fifo_output[1].write(upsampled1);
        }
    }
}


void maxpool(
    tapa::istreams<int16_v16, 2>& fifo_input,
    tapa::ostream<int16_v16>& fifo_output
) {
    maxpool_loop: for (;;) {
        ap_int<16> pkt[16];
        #pragma HLS array_partition variable=pkt complete
        maxpool_xvec: for (int j = 0; j < 2;) {
            #pragma HLS pipeline II=1 style=stp
            if(!fifo_input[0].empty() && !fifo_input[1].empty()){
                int16_v16 tmp0; fifo_input[0].try_read(tmp0);
                int16_v16 tmp1; fifo_input[1].try_read(tmp1);
                maxpool_x: for (int k = 0; k < 8; k++) {
                    #pragma HLS unroll
                    ap_int<16> a = (tmp0[2*k] > tmp1[2*k]) ? tmp0[2*k] : tmp1[2*k];
                    ap_int<16> b = (tmp0[2*k + 1] > tmp1[2*k + 1]) ? tmp0[2*k + 1] : tmp1[2*k + 1];
                    ap_int<16> fin_d = (a > b) ? a : b;
                    pkt[j*8 + k] = fin_d;
                }
                j++;
            }
        }
        int16_v16 send;
        for(int i = 0; i < 16; i++){
            #pragma HLS unroll
            send[i] = pkt[i];
        }
        fifo_output.write(send);
    }
}

void measure_cycle(tapa::istreams<bool, 3>& fifo_fin, tapa::mmap<int> cycle_count){
    for(int cycle = 0;;cycle++){
        bool flag = false;
        for(int i = 0; i < 3; i++){
            flag |= fifo_fin[i].empty();
        }
        if(!flag){
            for(int i = 0; i < 3; i++){
                fifo_fin[i].read(nullptr);
            }
            cycle_count[0] = cycle;
            break;
        }
    }
}


void CNN4L(
    tapa::mmap<int16_v16> X,
    tapa::mmap<int16_v16> kernel1,
    tapa::mmap<int16_v16> kernel2,
    tapa::mmap<int16_v16> kernel3,
    tapa::mmaps<int16_v16, 3> data_out,
    tapa::mmap<int> cycle_count
){
    tapa::stream<int16_v16, 16> fifo_X("fifo_X");
    tapa::stream<int16_v16> fifo_kernel1("fifo_kernel1");
    tapa::stream<int16_v16> fifo_kernel2("fifo_kernel2");
    tapa::stream<int16_v16> fifo_kernel3("fifo_kernel3");

    tapa::stream<int16_v16> fifo_conv1("fifo_conv1");
    tapa::stream<int16_v16> fifo_relu1("fifo_relu1");
    tapa::streams<int16_v32, 2> fifo_upsample1("fifo_upsample1");

    tapa::stream<int16_v16> fifo_conv2("fifo_conv2");
    tapa::stream<int16_v16> fifo_relu2("fifo_relu2");
    tapa::streams<int16_v32, 2> fifo_upsample2("fifo_upsample2");

    tapa::streams<int16_v16, 2> fifo_conv3("fifo_conv3");
    tapa::stream<int16_v16> fifo_maxpool3("fifo_maxpool3");
    tapa::stream<int16_v16> fifo_relu3("fifo_relu3");

    tapa::streams<int16_v16, 3> fifo_from_ctrl("fifo_from_ctrl");
    tapa::streams<int16_v16, 3> fifo_output("fifo_output");

    tapa::streams<bool, 3> fifo_fin("fifo_fin");

    tapa::task()
        .invoke<tapa::join>(
            read_X,
            X,
            fifo_X
        )
        .invoke<tapa::join>(
            read_W,
            2,
            kernel1,
            fifo_kernel1
        )
        .invoke<tapa::join>(
            read_W,
            2,
            kernel2,
            fifo_kernel2
        )
        .invoke<tapa::join>(
            read_W,
            2,
            kernel3,
            fifo_kernel3
        )

        // Layer 1
        .invoke<tapa::join>(
            CC0_Conv1_Conv4,
            fifo_X,
            fifo_kernel1,
            fifo_conv1,
            fifo_from_ctrl,
            fifo_output
        )
        .invoke<tapa::detach>(
            relu,
            fifo_conv1,
            fifo_relu1
        )
        .invoke<tapa::detach>(
            upsample,
            fifo_relu1,
            fifo_upsample1
        )

        // Layer 2
        .invoke<tapa::join>(
            CC1_Conv2_Conv4,
            fifo_upsample1,
            fifo_kernel2,
            fifo_conv2,
            fifo_from_ctrl,
            fifo_output
        )
        .invoke<tapa::detach>(
            relu,
            fifo_conv2,
            fifo_relu2
        )
        .invoke<tapa::detach>(
            upsample,
            fifo_relu2,
            fifo_upsample2
        )

        // Layer 3
        .invoke<tapa::join>(
            CC2_Conv3_Conv4,
            fifo_upsample2,
            fifo_kernel3,
            fifo_conv3,
            fifo_from_ctrl,
            fifo_output
        )
        .invoke<tapa::detach>(
            maxpool,
            fifo_conv3,
            fifo_maxpool3
        )
        .invoke<tapa::detach>(
            relu,
            fifo_maxpool3,
            fifo_relu3
        )
        // central mem
        .invoke<tapa::join>(
            central_mem,
            fifo_relu3,
            fifo_from_ctrl
        )
        .invoke<tapa::join, 3>(
            write_mtx,
            data_out,
            fifo_output,
            fifo_fin
        )
        .invoke<tapa::join>(
            measure_cycle,
            fifo_fin,
            cycle_count
        );
}