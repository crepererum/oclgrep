/* defines (see host code for documentation):
    - CACHE_MASK
    - FLAG_ITER_MAX
    - FLAG_STACK_FULL
    - GROUP_SIZE
    - ID_BEGIN
    - ID_FAIL
    - ID_OK
    - MAX_ITER_COUNT
    - MAX_STACK_SIZE
    - OVERSIZE_CACHE
    - RESULT_FAIL
    - SYNC_COUNT
    - USE_CACHE
*/

bool is_master() {
    return get_local_id(0) == 0;
}

uint find_next_slot(uint state, uint element, uint n, uint o, __constant uint* automatonData) {
    const uint base_node = automatonData[state] >> 2; // be careful about bytes vs indices!
    const uint m = automatonData[base_node];
    const uint base_node_body = base_node + 1;

    if (m == 0) {
        return 0;
    }

    bool found = false;

    uint idx_next = 0;
    uint base_next = base_node_body + idx_next * (1 + o);
    uint x_next = automatonData[base_next];

    uint idx_current = idx_next; // exception for first round
    uint base_current = base_next;
    uint x_current = x_next;

    while (idx_current + 1 < m && !found) {
        idx_current = idx_next;
        base_current = base_next;
        x_current = x_next;

        idx_next = idx_current + 1;
        base_next = base_node_body + idx_next * (1 + o);
        x_next = automatonData[base_next];

        if (element >= x_current && element < x_next) {
            found = true;
        }
    }

    if (found) {
        uint base_slot = base_current + 1;
        return base_slot;
    } else {
        return 0;
    }
}

uint state_from_slot(uint idx, uint base_slot, uint n, __constant uint* automatonData) {
    uint base_entry = base_slot + idx;
    uint next_state = automatonData[base_entry];
    if (next_state < n) {
        return next_state;
    } else {
        return ID_FAIL;
    }
}

struct stack_entry {
    uint startpos;
    uint pos;
    uint state;
};

bool sync(__local uint* active_count, uint iter_count, uint pos, __local uint* cache, __local uint* base_cache, __global const uint* text, uint size) {
    // do not sync every cycle
    if (iter_count % SYNC_COUNT == 0) {
        // use super safe double barrier
        barrier(CLK_LOCAL_MEM_FENCE);
        uint current_active_count = *active_count;
        if (is_master()) {
            *base_cache = 0xffffffff; // set base to MAX
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (current_active_count > 0) {
            // set new base to minimum (i.e. slowest thread)
            atomic_min(base_cache, pos);
            barrier(CLK_LOCAL_MEM_FENCE);

            // align cache base
            uint base_cache_aligned = *base_cache & (uint)(CACHE_MASK);

            // load cache
            for (uint i = 0; i < OVERSIZE_CACHE; ++i) {
                uint idx_cache = i * GROUP_SIZE + get_local_id(0);
                uint idx_text = base_cache_aligned + idx_cache;
                if (idx_text < size) {
                    cache[idx_cache] = text[idx_text];
                }
            }

            // master writes back aligned cache base
            if (is_master()) {
                *base_cache = base_cache_aligned;
            }

            // final barrier
            barrier(CLK_LOCAL_MEM_FENCE);

            return true;
        } else {
            return false;
        }
    } else {
        // if no check happened, assume there is still work to do
        return true;
    }
}

uint get_element(uint pos, __local uint* cache, __local uint* cache_base, __global const uint* text) {
#if USE_CACHE == 0
    return text[pos];
#else
    if (pos >= *cache_base && pos < (*cache_base + GROUP_SIZE * OVERSIZE_CACHE)) {
        return cache[pos - *cache_base];
    } else {
        return text[pos];
    }
#endif
}

__kernel void automaton(uint n,
                        uint o,
                        uint size,
                        uint multi_input_n,
                        __constant uint* automatonData,
                        __global const uint* text,
                        __global uint* output,
                        __global char* flags,
                        __local uint* cache) {
    // constants
    const uint base_group = get_group_id(0) * multi_input_n * GROUP_SIZE;

    // private thread-local state
    // WARNING: the stack is only supposed to hold valid tasks!
    //          (no ID_OK or ID_FAIL, only valid pos and startpos)
    __private struct stack_entry stack[MAX_STACK_SIZE];
    uint stack_size = 0;
    uint iter_count = 0;

    // cache preparation
    __local uint active_count;
    __local uint base_cache;
    if (is_master()) {
        active_count = GROUP_SIZE;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // push initial work to stack
    for (uint i = multi_input_n; i > 0; --i) {
        // [group_0: [i_0: [thread_0|...|thread_z] | ... | [i_y: thread_0|...|thread_z]]]
        // | ... |
        // [group_x: [i_0: [thread_0|...|thread_z] | ... | [i_y: thread_0|...|thread_z]]]
        uint startpos = base_group + (i - 1) * GROUP_SIZE + get_local_id(0);

        if (startpos < size) {
            // push to stack
            stack[stack_size].startpos = startpos;
            stack[stack_size].pos = startpos;
            stack[stack_size].state = ID_BEGIN;
            stack_size += 1;

            // write failed state, in case no task will finish
            output[startpos] = RESULT_FAIL;
        }
    }
    if (stack_size == 0) {
        atomic_dec(&active_count);
    }

    // stack is ordered, so same startpos values will be grouped together
    // use this knowledge for branch pruning in case we ware successful
    uint prune = 0xffffffff;

    // run until stack is empty
    bool work_left = true;
    uint pos_for_cache = 0xffffffff;
    while (work_left && iter_count < MAX_ITER_COUNT) {
        // 1. sync
#if USE_CACHE == 0
        work_left = stack_size > 0;
#else
        work_left = sync(&active_count, iter_count, pos_for_cache, cache, &base_cache, text, size);
#endif

        // 2. do thread-local work
        if (stack_size > 0) {
            // pop from stack
            stack_size -= 1;
            uint startpos = stack[stack_size].startpos;
            uint pos = stack[stack_size].pos;
            uint state = stack[stack_size].state;

            // pruning, see description above
            if (startpos != prune) {
                // run automaton one step
                uint element = get_element(pos, cache, &base_cache, text);
                uint base_slot = find_next_slot(state, element, n, o, automatonData);

                // decide what to do next
                if (base_slot != 0) {
                    // first decision gues to this "thread"
                    state = state_from_slot(0, base_slot, n, automatonData);

                    // all other possibilites go to the stack
                    for (uint i = 0; i < o; ++i) {
                        uint state_for_stack = state_from_slot(i, base_slot, n, automatonData);

                        // finished?
                        if (state_for_stack == ID_OK) {
                            // write output
                            output[startpos] = startpos;

                            // enable pruning
                            prune = startpos;

                            // remaining slot entries are not required
                            // HINT: regex compiler should sort slot entries so we do not push
                            //       additional (useless) data to the stack
                            break;
                        }

                        uint new_pos = pos + 1;
                        if (state_for_stack != ID_FAIL && new_pos < size) {
                            if (stack_size < MAX_STACK_SIZE) {
                                // push state to stack
                                stack[stack_size].startpos = startpos;
                                stack[stack_size].pos = new_pos;
                                stack[stack_size].state = state_for_stack;
                                stack_size += 1;
                                pos_for_cache = new_pos; // request pos from stack.top to be cached
                            } else {
                                flags[FLAG_STACK_FULL] = 1;
                            }
                        }
                    }
                }
            }

            // check if stack is empty => goodbye
            if (stack_size == 0) {
                atomic_dec(&active_count);
                pos_for_cache = 0xffffffff; // do not request any cache data anymore
            }
        }

        // continue counting
        iter_count += 1;
    }

    // write global error state
    if (iter_count >= MAX_ITER_COUNT) {
        flags[FLAG_ITER_MAX] = 1;
    }
}
