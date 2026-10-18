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
from http.cookiejar import request_port
from select import select

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

    @staticmethod
    def report(bundles):
        n = sum(1 for b in bundles if set(b) != {"debug"})
        for engine, limit in SLOT_LIMITS.items():
            if engine == "debug":
                continue
            used = sum(len(b.get(engine, [])) for b in bundles)
            print(f"{engine:5s}: {used:6d} ops {used/n:5.2f}/cycle of {limit}")

    @staticmethod
    def show(bundles, start = 0, count=20):
        for cycle in range(start, min(start+count, len(bundles))):
            bundle = bundles[cycle]
            print(f"cycle: {cycle:5d}: " + " ".join(
                f"{engine}={len(bundle.get(engine,[]))}" for engine in SLOT_LIMITS.keys()
            ))

    def scheduler_fcfs(self, ops):
        schedule = []
        available_at = {}
        write_after_write = {}
        write_after_read = {}
        for engine, instr in ops:
            cycle = 0
            reads, writes = self.get_operands(engine, instr)

            # RAW: Are all inputs available? If not, when are they available?
            for r in reads:
                cycle = max(available_at.get(r, 0), cycle)
            # WAW and WAR: Is it safe to write to w at this cycle? If not, when can we override the w?
            for w in writes:
                cycle = max(write_after_write.get(w, 0), cycle)
                cycle = max(write_after_read.get(w, 0), cycle)

            # In what cycle the resource is available to schedule this instr?
            while True:
                while len(schedule) <= cycle:
                    schedule.append({})
                if len(schedule[cycle].get(engine, [])) < SLOT_LIMITS[engine]:
                    break
                cycle += 1

            # Schedule the operation in this @cycle
            schedule[cycle].setdefault(engine, []).append(instr)

            # Record
            for w in writes:
                # RAW: w value can be readable at cycle + 1, after instr complete at cycle
                available_at[w] = cycle + 1
                # WAW: similarly, we can write to w value at cycle + 1
                write_after_write[w] = cycle + 1
            for r in reads:
                # WAR: we can write after read in the same cycle, because read happens at the
                # start of the cycle. But there are other instruction reading operands. We
                # can only write after all read are finished. Hence,
                # we need to consider max for each read.
                write_after_read[r] = max(write_after_read.get(r, 0), cycle)

        return schedule


    @staticmethod
    def get_operands(engine, instr):
        mem = "MEM"

        """
        return [read operands], [write_operands]
        """
        def vector(base):
            return range(base, base + VLEN)

        match engine, instr:
            # ALU
            case "alu", (_, dest, a, b):
                return [a, b], [dest]

            # VALU - Vector Ops
            case "valu", ("vbroadcast", dest, src):
                return [src], [*vector(dest)]
            case "valu", ("multiply_add", dest, a, b, c):
                return [*vector(a), *vector(b), *vector(c)], [*vector(dest)]
            case "valu", (_, dest, a, b):
                return [*vector(a), *vector(b)], [*vector(dest)]

            # LOAD
            case "load", ("const", dest, _):
                return [], [dest]
            case "load", ("load", dest, addr):
                return [addr, mem], [dest]
            case "load", ("load_offset", dest, src, offset):
                return [src + offset, mem], [dest + offset]
            case "load", ("vload", dest, addr):
                return [addr, mem], [*vector(dest)]

            # STORE
            case "store", ("store", addr, src):
                return [src, addr], [mem]
            case "store", ("vstore", addr, src):
                return [addr, *vector(src)], [mem]

            # FLOW
            case "flow", ("select", dest, cond, a, b):
                return [a, b, cond], [dest]
            case "flow", ("vselect", dest, cond, a, b):
                return [*vector(a), *vector(b), *vector(cond)], [*vector(dest)]
            case "flow", ("add_imm", dest, a, imm_val):
                return [a], [dest]

            # DEBUG
            case "debug", ("compare", loc, _key):
                return [loc], []
            case "debug", ("vcompare", loc, v_keys):
                return [*vector(loc)], []
            case "debug", _:
                return [], []

        raise NotImplementedError(f"Instructions unhandled: {engine}: {instr}")

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

        n_blocks:int  = batch_size // VLEN
        # Vector scratch registers for each block
        # batch_size // VLEN = 256 // 8 = 32 blocks
        # 32 * 32 = 1024 scratch cells
        regs = [{
            "idx" : self.alloc_scratch(f"idx_{block}", 8),
            "val" : self.alloc_scratch(f"val_{block}", 8),
            "node" : self.alloc_scratch(f"node_{block}", 8),
            "vtmp1": self.alloc_scratch(f"tmp_{block}", 8)}
            for block in range(n_blocks)
        ]

        # Broadcast inp_indices_p, inp_values_p and forest_values_p pointers and n_nodes
        v_n_nodes = self.alloc_scratch("v_n_nodes", 8)
        self.add("valu", ("vbroadcast", self.scratch["v_n_nodes"], self.scratch["n_nodes"]))

        v_forest_values_p = self.alloc_scratch("v_forest_values_p", 8)
        self.add("valu", ("vbroadcast", self.scratch["v_forest_values_p"], self.scratch["forest_values_p"]))

        for b, r in enumerate(regs):
            block_start = b * VLEN
            block = self.scratch_const(block_start)
            # idx = mem[inp_indices_p + block_start]
            body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], block)))
            body.append(("load", ("vload", r["idx"], tmp_addr)))
            body.append(("debug", ("vcompare", r["idx"], self.debug_vkeys(0, block_start, "idx"))))

            # val = mem[inp_values_p + block_start]
            body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], block)))
            body.append(("load", ("vload", r["val"], tmp_addr)))
            body.append(("debug", ("vcompare", r["val"], self.debug_vkeys(0, block_start, "val"))))

        for round in range(rounds):
            for b, r in enumerate(regs):
                block_start = b * VLEN

                # node_val = mem[forest_values_p + idx], gather op
                # we can't use one pointer
                body.append(("valu", ("+", r["node"], v_forest_values_p, r["idx"])))
                for offset in range(8):
                    # scratch[node + offset] = mem[scratch[node + offset]]
                    body.append(("load", ("load_offset", r["node"], r["node"], offset)))
                body.append(("debug", ("vcompare", r["node"], self.debug_vkeys(round, block_start, "node_val"))))

                # val = myhash(val ^ node_val)
                body.append(("valu", ("^", r["val"], r["val"], r["node"])))
                # Node value are not used after the above op, we can re-use it as tmp2
                body.extend(self.build_hash(r["val"], r["vtmp1"], r["node"], round, block_start))
                body.append(("debug", ("vcompare", r["val"], self.debug_vkeys(round, block_start, "hashed_val"))))

                # idx = 2*idx + (1 if val % 2 == 0 else 2) simplified to the following
                # bit = val & 1
                # 2*idx + 1 + bit
                body.append(("valu", ("multiply_add", r["idx"], r["idx"], self.scratch_vconst(2), self.scratch_vconst(1))))
                body.append(("valu", ("&", r["vtmp1"], r["val"], self.scratch_vconst(1))))
                body.append(("valu", ("+", r["idx"], r["idx"], r["vtmp1"])))

                # idx = 0 if idx >= n_nodes else idx
                body.append(("valu", ("<", r["vtmp1"], r["idx"], v_n_nodes)))
                body.append(("flow", ("vselect", r["idx"], r["vtmp1"], r["idx"], self.scratch_vconst(0))))
                body.append(("debug", ("vcompare", r["idx"], self.debug_vkeys(round, block_start, "wrapped_idx"))))

        for b, r in enumerate(regs):
            block = self.scratch_const(b * VLEN)
            # mem[inp_indices_p + block_idx * VLEN] = idx
            # body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], block)))
            # body.append(("store", ("vstore", tmp_addr, r["idx"])))

            # mem[inp_values_p + block_idx * VLEN] = val
            body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], block)))
            body.append(("store", ("vstore", tmp_addr, r["val"])))

        body_instrs = self.build(body)
        print("\nReport before scheduling: \n")
        self.report(body_instrs)
        self.show(body_instrs, 0, 20)

        body_instrs = self.scheduler_fcfs(body)
        print("\nReport after scheduling: \n")
        self.report(body_instrs)
        self.show(body_instrs, 0, 20)

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
