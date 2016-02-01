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

__constant uint* find_next_slot(uint state, uint element, uint n, uint o, __constant uint* automatonData) {
    const uint base_node = automatonData[state] >> 2; // be careful about bytes vs indices!
    __constant uint* pNode = automatonData + base_node;
    const uint m = pNode[0];
    __constant uint* pNodeBody = pNode + 1;

    if (m == 0) {
        return 0;
    }

    bool searching = true;

    __constant uint* pNext = pNodeBody;
    uint x_next = pNodeBody[0];

    // exception for first round
    __constant uint* pCurrent = pNext;
    uint x_current = x_next;

    __constant uint* pPreEnd = pNodeBody + (m - 1) * (1 + o);

    while (pNext < pPreEnd && searching) {
        pCurrent = pNext;
        x_current = x_next;

        pNext += (1 + o);
        x_next = *pNext;

        if (element >= x_current && element < x_next) {
            searching = false;
        }
    }

    if (searching) {
        return 0;
    } else {
        return pCurrent + 1;
    }
}

uint state_from_slot(uint idx, __constant uint* pSlot, uint n) {
    uint next_state = pSlot[idx];
    if (next_state < n) {
        return next_state;
    } else {
        return ID_FAIL;
    }
}

struct stack_entry {
    uint pos;
    uint state;
};

#if USE_CACHE != 0
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
#endif

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
    uint input_round = 0;

    // cache preparation
    __local uint active_count;
    __local uint base_cache;
    if (is_master()) {
        active_count = GROUP_SIZE;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // run until stack is empty
    bool work_left = true;
    uint pos_for_cache = 0xffffffff;
    uint startpos = 0xffffffff; // will never be used
    while (work_left && iter_count < MAX_ITER_COUNT) {
        // 1. refill
        if (stack_size == 0 && input_round < multi_input_n) {
            startpos = base_group + input_round * GROUP_SIZE + get_local_id(0);

            if (startpos < size) {
                // push to stack
                stack[0].pos = startpos;
                stack[0].state = ID_BEGIN;
                stack_size = 1;
                pos_for_cache = startpos;

                // write failed state, in case no task will finish
                output[startpos] = RESULT_FAIL;
            }

            input_round += 1;
            if (input_round == multi_input_n) {
                atomic_dec(&active_count);
                pos_for_cache = 0xffffffff; // do not request any cache data anymore
            }
        }

        // 2. sync
#if USE_CACHE == 0
        work_left = stack_size > 0 || input_round < multi_input_n;
#else
        work_left = sync(&active_count, iter_count, pos_for_cache, cache, &base_cache, text, size);
#endif

        // 3. do thread-local work
        if (stack_size > 0) {
            // pop from stack
            stack_size -= 1;
            uint pos = stack[stack_size].pos;
            uint state = stack[stack_size].state;

            // run automaton one step
            uint element = get_element(pos, cache, &base_cache, text);
            __constant uint* pSlot = find_next_slot(state, element, n, o, automatonData);

            // decide what to do next
            if (pSlot) {
                // new data for stack
                bool not_finished = true;
                for (uint i = 0; i < o && not_finished; ++i) {
                    uint state_for_stack = state_from_slot(i, pSlot, n);
                    uint new_pos = pos + 1;

                    // finished?
                    if (state_for_stack == ID_OK) {
                        // write output
                        output[startpos] = startpos;

                        // prune remaining data
                        stack_size = 0;

                        // remaining slot entries are not required
                        not_finished = false;
                    } else if (state_for_stack != ID_FAIL && new_pos < size) {
                        if (stack_size < MAX_STACK_SIZE) {
                            // push state to stack
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

        // 4. continue counting
        iter_count += 1;
    }

    // write global error state
    if (iter_count >= MAX_ITER_COUNT) {
        flags[FLAG_ITER_MAX] = 1;
    }
}
