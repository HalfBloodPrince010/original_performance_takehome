"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vconst_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def debug_vkeys(self, round, i, name, *extra):
        return [(round, i + j, name, *extra) for j in range(8)]

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def scratch_vconst(self, val, name=None):
        if val not in self.vconst_map:
            src_val_scratch_addr = self.scratch_const(val)
            addr = self.alloc_scratch(name, 8)
            self.add("valu", ("vbroadcast", addr, src_val_scratch_addr))
            self.vconst_map[val] = addr
        return self.vconst_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        # Stage 0 - ("+", 0x7ED55D16, "+", "<<", 12) folded to ("*", 4097, "+", 0x7ED55D16)
        # val << x = val * 2^x
        slots.append(("valu", ("multiply_add", val_hash_addr, val_hash_addr, self.scratch_vconst(4097), self.scratch_vconst(0x7ED55D16))))
        slots.append(("debug", ("vcompare", val_hash_addr, self.debug_vkeys(round, i, "hash_stage", 0))))

        # Stage 1 - ("^", 0xC761C23C, "^", ">>", 19)
        slots.append(("valu", ("^", tmp1, val_hash_addr, self.scratch_vconst(0xC761C23C))))
        slots.append(("valu", (">>", tmp2, val_hash_addr, self.scratch_vconst(19))))
        slots.append(("valu", ("^", val_hash_addr, tmp1, tmp2)))
        slots.append(("debug", ("vcompare", val_hash_addr, self.debug_vkeys(round, i, "hash_stage", 1))))

        # Stage 2 - ("+", 0x165667B1, "+", "<<", 5)
        # Stage 2 folded & merged with 3
        # Stage 3 - ("+", 0xD3A2646C, "^", "<<", 9)
        slots.append(("valu", ("multiply_add", tmp1, val_hash_addr, self.scratch_vconst(33), self.scratch_vconst(0xE9F8CC1D))))
        slots.append(("valu", ("multiply_add", tmp2, val_hash_addr, self.scratch_vconst(16896), self.scratch_vconst(0xACCF6200))))
        slots.append(("valu", ("^", val_hash_addr, tmp1, tmp2)))
        slots.append(("debug", ("vcompare", val_hash_addr, self.debug_vkeys(round, i, "hash_stage", 3))))

        # Stage 4 - ("+", 0xFD7046C5, "+", "<<", 3)  folded to ("*", 9, "+", 0xFD7046C5)
        slots.append(("valu", ("multiply_add", val_hash_addr, val_hash_addr, self.scratch_vconst(9), self.scratch_vconst(0xFD7046C5))))
        slots.append(("debug", ("vcompare", val_hash_addr, self.debug_vkeys(round, i, "hash_stage", 4))))

        # Stage 5 - ("^", 0xB55A4F09, "^", ">>", 16)
        slots.append(("valu", ("^", tmp1, val_hash_addr, self.scratch_vconst(0xB55A4F09))))
        slots.append(("valu", (">>", tmp2, val_hash_addr, self.scratch_vconst(16))))
        slots.append(("valu", ("^", val_hash_addr, tmp1, tmp2)))
        slots.append(("debug", ("vcompare", val_hash_addr, self.debug_vkeys(round, i, "hash_stage", 5))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        Scalar implementation using only scalar ALU and load/store.
        """
        tmp1 = self.alloc_scratch("tmp1")

        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        body = []  # array of slots

        # Scalar scratch registers
        tmp_addr = self.alloc_scratch("tmp_addr")

        # Vector scratch registers
        tmp_idx_i = self.alloc_scratch("tmp_idx_i", 8)
        tmp_val_i = self.alloc_scratch("tmp_val_i", 8)
        tmp_node_val_i = self.alloc_scratch("tmp_node_val_i", 8)
        tmp_addr_i = self.alloc_scratch("tmp_addr_i", 8)

        # Vector tmp
        vtmp1 = self.alloc_scratch("vtmp1", 8)
        vtmp2 = self.alloc_scratch("vtmp2", 8)
        vtmp3 = self.alloc_scratch("vtmp3", 8)

        # Broadcast inp_indices_p, inp_values_p and forest_values_p pointers and n_nodes
        v_n_nodes = self.alloc_scratch("v_n_nodes", 8)
        self.add("valu", ("vbroadcast", self.scratch["v_n_nodes"], self.scratch["n_nodes"]))

        v_forest_values_p = self.alloc_scratch("v_forest_values_p", 8)
        self.add("valu", ("vbroadcast", self.scratch["v_forest_values_p"], self.scratch["forest_values_p"]))

        for round in range(rounds):
            for i in range(0, batch_size, 8):
                i_const = self.scratch_const(i)

                # idx = mem[inp_indices_p + i], loads are continuous from the tmp_addr. So, we don't need valu here.
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("load", ("vload", tmp_idx_i, tmp_addr)))
                body.append(("debug", ("vcompare", tmp_idx_i, self.debug_vkeys(round, i, "idx"))))

                # val = mem[inp_values_p + i]
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("load", ("vload", tmp_val_i, tmp_addr)))
                body.append(("debug", ("vcompare", tmp_val_i, self.debug_vkeys(round, i, "val"))))

                # node_val = mem[forest_values_p + idx], gather op
                # we can't use one pointer
                body.append(("valu", ("+", tmp_addr_i, v_forest_values_p, tmp_idx_i)))
                for offset in range(8):
                    # scratch[tmp_node_val_i + offset] = mem[scratch[tmp_addr_i + offset]]
                    body.append(("load", ("load_offset", tmp_node_val_i, tmp_addr_i, offset)))
                body.append(("debug", ("vcompare", tmp_node_val_i, self.debug_vkeys(round, i, "node_val"))))

                # val = myhash(val ^ node_val)
                body.append(("valu", ("^", tmp_val_i, tmp_val_i, tmp_node_val_i)))
                body.extend(self.build_hash(tmp_val_i, vtmp1, vtmp2, round, i))
                body.append(("debug", ("vcompare", tmp_val_i, self.debug_vkeys(round, i, "hashed_val"))))

                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                body.append(("valu", ("%", vtmp1, tmp_val_i, self.scratch_vconst(2))))
                body.append(("valu", ("==", vtmp1, vtmp1, self.scratch_vconst(0))))
                body.append(("flow", ("vselect", vtmp3, vtmp1, self.scratch_vconst(1), self.scratch_vconst(2))))
                body.append(("valu", ("*", tmp_idx_i, tmp_idx_i, self.scratch_vconst(2))))
                body.append(("valu", ("+", tmp_idx_i, tmp_idx_i, vtmp3)))

                # idx = 0 if idx >= n_nodes else idx
                body.append(("valu", ("<", vtmp1, tmp_idx_i, v_n_nodes)))
                body.append(("flow", ("vselect", tmp_idx_i, vtmp1, tmp_idx_i, self.scratch_vconst(0))))
                body.append(("debug", ("vcompare", tmp_idx_i, self.debug_vkeys(round, i, "wrapped_idx"))))

                # mem[inp_indices_p + i] = idx
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
                body.append(("store", ("vstore", tmp_addr, tmp_idx_i)))

                # mem[inp_values_p + i] = val
                body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
                body.append(("store", ("vstore", tmp_addr, tmp_val_i)))

        body_instrs = self.build(body)
        self.instrs.extend(body_instrs)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
