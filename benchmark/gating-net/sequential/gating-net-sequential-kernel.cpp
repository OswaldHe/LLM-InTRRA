#include <iostream>
#include <ap_int.h>
#include <tapa.h>
#include <hls_math.h>
// #include <glog/logging.h>

using namespace std;

const int B = 32;
const int ID = 4096;
const int HD = 11008;
const int VEC_LEN = 32;
const int SCALE_FACTOR = 32;
constexpr int ID_div_VEC_LEN = ID / VEC_LEN;
constexpr int HD_div_VEC_LEN = HD / VEC_LEN;
constexpr int B_div_VEC_LEN = B / VEC_LEN;

typedef ap_int<16> type_t;
using vec_t = tapa::vec_t<type_t, VEC_LEN>;
constexpr int input_size = B * ID / VEC_LEN;

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

void vector_mac(const vec_t& a, const vec_t& b, type_t& c){
#pragma HLS inline
    for (int i = 0; i < VEC_LEN; i++){
        c += a[i] * b[i];
    }
}

/**
 * @brief Top function for gating net
 * 
 * @param input 
 * @param W_up 
 * @param W_gate 
 * @param W_down 
 * @param combined HBM For combined result of up and gate
 * @param output 
 * @param fifo_fin 
 */
void gating_net_top(
    tapa::async_mmap<vec_t>& input,
    tapa::async_mmap<vec_t>& W_up,
    tapa::async_mmap<vec_t>& W_gate,
    tapa::async_mmap<vec_t>& W_down,
    tapa::async_mmap<vec_t>& combined,
    tapa::async_mmap<vec_t>& output,
    tapa::ostream<bool>& fifo_fin
) {
    vec_t input_cache[B][ID_div_VEC_LEN];
    vec_t col_W_up[ID_div_VEC_LEN];
    vec_t col_W_gate[ID_div_VEC_LEN];
    vec_t up_result; type_t up_result_tmp[B];
    vec_t tmp_out[B];
    vec_t tmp_vec;

#pragma HLS ARRAY_PARTITION variable=input_cache type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=up_result_tmp type=complete dim=1

    // LOG(INFO) << "Start reading input";

    // Read and cache input
    read_input: for(int i_req = 0, i_resp = 0; i_resp < input_size;) {
        if((i_req < input_size) & !input.read_addr.full()) {
            input.read_addr.write(i_req);
            i_req++;
        }
        bool success = input.read_data.try_read(tmp_vec);
        if(success) {
            input_cache[i_resp / ID_div_VEC_LEN][i_resp % ID_div_VEC_LEN] = tmp_vec;
            i_resp++;
        }
    }

    // LOG(INFO) << "Start computing up and product with gate";

    // Compute up and product with gate
    combined_projection: for (int j = 0; j < HD;) {
        // readin a column of W_up
        read_W_up: for (int k_req = 0, k_resp = 0; k_resp < ID_div_VEC_LEN;) {
            if (k_req < ID_div_VEC_LEN && !W_up.read_addr.full()) {
                W_up.read_addr.write(j * ID_div_VEC_LEN + k_req);
                k_req++;
            }
            bool success = W_up.read_data.try_read(col_W_up[k_resp]);
            if(success) {
                k_resp++;
            }
        }

        // LOG(INFO) << "Read W_up " << j << " done";

        // readin a column of W_gate
        read_W_gate: for (int k_req = 0, k_resp = 0; k_resp < ID_div_VEC_LEN;) {
            if (k_req < ID_div_VEC_LEN && !W_gate.read_addr.full()) {
                W_gate.read_addr.write(j * ID_div_VEC_LEN + k_req);
                k_req++;
            }
            bool success = W_gate.read_data.try_read(col_W_gate[k_resp]);
            if(success) {
                k_resp++;
            }
        }

        // LOG(INFO) << "Read W_gate " << j << " done";

        // compute a column
        compute_combined: for (int i = 0; i < B; i++) {
#pragma HLS UNROLL factor=8
            type_t acc = 0;
            type_t gate_acc = 0;
            compute_combined_k: for (int k = 0; k < ID_div_VEC_LEN; k++) {
#pragma HLS PIPELINE II=1 style=stp
                vec_t input_seg = input_cache[i][k];
                for (int kk = 0; kk < VEC_LEN; kk++) {
                    acc += input_seg[kk] * col_W_up[k][kk];
                    gate_acc += input_seg[kk] * col_W_gate[k][kk];
                }
            }
            // up_result_tmp[i] = acc / (1 + (type_t) hls::exp(-acc)) * gate_acc;
            up_result_tmp[i] = acc * gate_acc;
        }

        // LOG(INFO) << "Computed combined " << j << " done";

        for (int i = 0; i < B; i++) {
#pragma HLS UNROLL
            up_result[i] = up_result_tmp[i];
        }

        // write the column to combined
        write_combined_col: for (int i_req = 0, i_resp = 0; i_resp < 1;) {
            if (i_req < B_div_VEC_LEN && !combined.write_addr.full() && !combined.write_data.full()) {
                combined.write_addr.write(j);
                combined.write_data.try_write(up_result);
                i_req++;
            }
            bool success = false;
            auto resp = combined.write_resp.read(success);
            if(success) {
                i_resp += unsigned(resp)+1;
            }
        }

        // LOG(INFO) << "Write combined " << j << " done";

        j++;
    }

    // LOG(INFO) << "Computed combined";

    clear_input_cache: for (int i = 0; i < B; i++) {
        for (int j = 0; j < ID_div_VEC_LEN; j++) {
            for (int k = 0; k < VEC_LEN; k++) {
#pragma HLS UNROLL
                input_cache[i][j][k] = 0;
            }
        }
    }

    // LOG(INFO) << "Cleared input cache";

    // Down projection with outer product
    down_projection: for (int k = 0; k < HD; k++) {
        vec_t tmp_row[ID_div_VEC_LEN];
        vec_t tmp_col;
        type_t tmp_vals[B];

#pragma HLS ARRAY_PARTITION variable=tmp_vals type=complete dim=1

        // readin a row of W_down
        read_W_down: for (int j_req = 0, j_resp = 0; j_resp < ID_div_VEC_LEN;) {
            if (j_req < ID_div_VEC_LEN && !W_down.read_addr.full()) {
                W_down.read_addr.write(k * ID_div_VEC_LEN + j_req);
                j_req++;
            }
            bool success = W_down.read_data.try_read(tmp_row[j_resp]);
            if(success) {
                j_resp++;
            }
        }

        // readin a column of combined
        read_combined: for (int i_req = 0, i_resp = 0; i_resp < B_div_VEC_LEN;) {
            if (i_req < B_div_VEC_LEN && !combined.read_addr.full()) {
                combined.read_addr.write(k);
                i_req++;
            }
            bool success = combined.read_data.try_read(tmp_col);
            if(success) {
                for (int i = 0; i < B; i++) {
#pragma HLS UNROLL
                    tmp_vals[i] = tmp_col[i];
                }
                i_resp++;
            }
        }

        // compute
        compute_down_projection_tiles: for (int i = 0; i < B; i++) {
#pragma HLS UNROLL factor=8
            type_t col_val = tmp_vals[i];
            for (int j = 0; j < ID_div_VEC_LEN; j++) {      
#pragma HLS LOOP_TRIPCOUNT min=ID_div_VEC_LEN max=ID_div_VEC_LEN
#pragma HLS PIPELINE II=1 style=stp
                vec_t tmp_vec = input_cache[i][j];
                vec_t local_tmp_row = tmp_row[j];
                for (int jj = 0; jj < VEC_LEN; jj++) {
#pragma HLS UNROLL
                    tmp_vec[jj] = tmp_vec[jj] + local_tmp_row[jj] * col_val;
                }
                input_cache[i][j] = tmp_vec;
            }
        }
        // LOG(INFO) << "Computed iteration " << k << " in down_projection with value " << double(input_cache[0][0]);
    }

    // LOG(INFO) << "Computed down projection";

    // Write output tile by tile
    write_output: for (int i = 0; i < B; i++) {
        for (int j = 0; j < ID_div_VEC_LEN; j++) {
            for (int i_req = 0, i_resp = 0; i_resp < ID_div_VEC_LEN;) {
                if (i_req < ID_div_VEC_LEN && !output.write_addr.full() && !output.write_data.full()) {
                    output.write_addr.write(i * ID_div_VEC_LEN + j);
                    output.write_data.try_write(input_cache[i][j]);
                    i_req++;
                }
                bool success = false;
                auto resp = output.write_resp.read(success);
                if(success) {
                    i_resp += unsigned(resp)+1;
                    // // LOG(INFO) << "Write output " << "(" << i_resp << ")" << " in gating_net_top with value " << double(input_cache[i_resp][0]);
                }
            }
        }
    }

    // LOG(INFO) << "Kernel done";

    fifo_fin.write(true);
}

void gating_net(
    tapa::mmap<vec_t> top_input,
    tapa::mmap<vec_t> W_up,
    tapa::mmap<vec_t> W_gate,
    tapa::mmap<vec_t> W_down,
    tapa::mmap<vec_t> combined,
    tapa::mmap<vec_t> top_output,
    tapa::mmap<int> cycle_count
) {
    tapa::streams<bool, 1> fifo_fin("fifo_fin");

    tapa::task()
        .invoke<tapa::join>(
            gating_net_top,
            top_input,
            W_up,
            W_gate,
            W_down,
            combined,
            top_output,
            fifo_fin
        )
        .invoke<tapa::join>(
            measure_cycle,
            fifo_fin,
            cycle_count
        )
    ;
}
