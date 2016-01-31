#define ID_FAIL 0
#define ID_OK 1
#define ID_BEGIN 2
#define RESULT_FAIL 0xffffffff
#define MAX_STACK_SIZE 16

uint find_next_slot(uint state, uint element, uint n, uint m, uint o, __constant char* automatonData) {
    uint base_node = state * m * (sizeof(uint) + o * sizeof(uint));

    bool found = false;

    uint idx_next = 0;
    uint base_next = base_node + idx_next * (sizeof(uint) + o * sizeof(uint));
    uint x_next = *((__constant uint*)(automatonData + base_next));

    uint idx_current = idx_next; // exception for first round
    uint base_current = base_next;
    uint x_current = x_next;

    while (idx_current + 1 < m && !found) {
        idx_current = idx_next;
        base_current = base_next;
        x_current = x_next;

        idx_next = idx_current + 1;
        base_next = base_node + idx_next * (sizeof(uint) + o * sizeof(uint));
        x_next = *((__constant uint*)(automatonData + base_next));

        if (element >= x_current && element < x_next) {
            found = true;
        }
    }

    if (found) {
        uint base_slot = base_current + sizeof(uint);
        return base_slot;
    } else {
        return 0;
    }
}

uint state_from_slot(uint idx, uint base_slot, uint n, __constant char* automatonData) {
    uint base_entry = base_slot + idx * sizeof(uint);
    uint next_state = *((__constant uint*)(automatonData + base_entry));
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

uint run_until_end(uint pos, uint state, uint n, uint m, uint o, uint size, __constant char* automatonData, __global const char* text, __private struct stack_entry* stack, uint* stack_size) {
    while (state != ID_FAIL && state != ID_OK && pos < size) {
        // evaluate current node
        uint element = *((__global const uint*)(text + sizeof(uint) * pos));
        uint base_slot = find_next_slot(state, element, n, m, o, automatonData);
        pos += 1;

        // decide what to do next
        if (base_slot != 0) {
            // first decision gues to this "thread"
            state = state_from_slot(0, base_slot, n, automatonData); // XXX: push other states to stack

            // all other possibilites go to the stack
            for (uint i = 1; i < o; ++i) {
                uint state_for_stack = state_from_slot(i, base_slot, n, automatonData);

                // XXX: warn user when stack was full
                if (state_for_stack != ID_FAIL && *stack_size < MAX_STACK_SIZE) {
                    // push state to stack
                    stack[*stack_size].pos = pos;
                    stack[*stack_size].state = state_for_stack;
                    *stack_size += 1;
                }
            }
        } else {
            state = ID_FAIL;
        }
    }

    return state;
}

__kernel void automaton(uint n, uint m, uint o, uint size, __constant char* automatonData, __global const char* text, __global uint* output) {
    uint startpos = get_global_id(0);
    if (startpos < size) {
        // set up stack with inital task
        __private struct stack_entry stack[MAX_STACK_SIZE];
        uint stack_size = 1;
        stack[0].pos = startpos;
        stack[0].state = ID_BEGIN;

        // run until stack is empty
        uint result_state = ID_FAIL;
        while (stack_size > 0 && result_state != ID_OK) {
            // pop from stack
            stack_size -= 1;
            uint pos = stack[stack_size].pos;
            uint state = stack[stack_size].state;

            // run automaton until end and check if that branch was successfull
            uint candidate = run_until_end(pos, state, n, m, o, size, automatonData, text, stack, &stack_size);
            if (candidate == ID_OK) {
                result_state = ID_OK;
            }
        }

        // check for final result
        if (result_state == ID_OK) {
            output[startpos] = startpos;
        } else {
            output[startpos] = RESULT_FAIL;
        }
    }
}
