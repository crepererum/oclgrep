/* defines (see host code for documentation):
    - RESULT_FAIL
*/

__kernel void transform(__global const uint* in, __global uint* out, uint size) {
    uint idx = get_global_id(0);
    if (idx < size) {
        uint x = in[idx];
        if (x != RESULT_FAIL) {
            out[idx] = 1;
        } else {
            out[idx] = 0;
        }
    }
}

__kernel void scan(const __global uint* in, __global uint* out, uint size, uint offset) {
    uint idx_right = get_global_id(0);
    if (idx_right < size) {
        uint right = in[idx_right];
        uint left = 0;
        if (idx_right >= offset) {
            uint idx_left = idx_right - offset;
            left = in[idx_left];
        }
        out[idx_right] = left + right;
    }
}

__kernel void move(__global const uint* in_rank, __global const uint* in_data, __global uint* out, uint size) {
    uint idx_source = get_global_id(0);
    if (idx_source < size) {
        uint rank = in_rank[idx_source];
        uint x = in_data[idx_source];
        if (x != RESULT_FAIL) {
            uint idx_target = rank - 1; // inclusive to exclusive
            out[idx_target] = x;
        }
    }
}
