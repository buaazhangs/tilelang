"""Independent Cube matrix views may reuse one physical allocation."""

import math

import pytest
import tilelang
import tilelang.testing
from tilelang import tvm
from tilelang.ascend import language as T, transform
from tilelang.layout import Layout, make_ascend_major_k_layout
from tvm import tirx
from tvm.tirx.script.builder.ir import sblock_attr
from tvm.tirx.stmt_functor import ir_transform

from testing.ascend._ir import calls, kernel, nodes, seq


INDEPENDENT_LAYOUT_BUFFERS = "tl.independent_layout_buffers"


def shared_l0b_program(unanchored_read=False):
    @T.prim_func
    def main(QK: T.Tensor((128, 128), "bfloat16"), PV: T.Tensor((512, 32), "bfloat16")):
        with T.Kernel(1):
            qk_l1 = T.alloc_l1((128, 128), "bfloat16")
            pv_l1 = T.alloc_l1((512, 32), "bfloat16")
            qk_l0b = T.alloc_l0b((128, 128), "bfloat16")
            pv_l0b = T.view(qk_l0b, (512, 32))
            qk_l0a = T.alloc_l0a((16, 128), "bfloat16")
            pv_l0a = T.alloc_l0a((16, 32), "bfloat16")
            accum = T.alloc_l0c((16, 512), "float32")
            T.copy(QK, qk_l1)
            T.copy(qk_l1, qk_l0b)
            T.gemm(qk_l0a, qk_l0b, accum[:, :128], transpose_B=True, clear_accum=True)
            T.copy(PV, pv_l1)
            T.copy(pv_l1, pv_l0b)
            T.gemm(pv_l0a, pv_l0b, accum, transpose_B=True, clear_accum=True)
            if unanchored_read:
                flat = T.view(qk_l0b, (16384,))
                T.evaluate(flat[16])

    return main


def frontend_module(program=None):
    target = tvm.target.Target("ascend")
    program = shared_l0b_program() if program is None else program
    mod = tvm.IRModule({"main": program.with_attr("target", target)})
    return tilelang.transform.MaterializeKernelLaunch()(mod)


def canonical_view_module(pv_shape=(512, 32)):
    qk = tirx.decl_buffer((128, 128), "bfloat16", name="qk", scope="shared.l0b")
    pv = tirx.decl_buffer(pv_shape, "bfloat16", name="pv", data=qk.data)
    body = seq(
        tirx.BufferStore(qk, tirx.const(1, "bfloat16"), [16, 16]),
        tirx.Evaluate(T.access_ptr(qk[16, 16], "r", extent=1)),
        tirx.BufferStore(pv, tirx.const(2, "bfloat16"), [16, 16]),
        tirx.Evaluate(T.access_ptr(pv[16, 16], "r", extent=1)),
    )
    return kernel(
        body,
        buffers=[qk],
        annotations={"layout_map": {qk: make_ascend_major_k_layout(qk), pv: make_ascend_major_k_layout(pv)}},
    )


def _layouts(mod):
    return {buffer: layout for block in nodes(mod, tirx.SBlock) for buffer, layout in block.annotations.get("layout_map", {}).items()}


def _assert_common_allocation(mod, views):
    allocations = [buffer for block in nodes(mod, tirx.SBlock) for buffer in block.alloc_buffers if buffer.scope() == views[0].scope()]
    assert len(allocations) == 1
    assert math.prod(int(dim) for dim in allocations[0].shape) * 2 == 32768
    assert all(view.data.same_as(allocations[0].data) for view in views)
    assert not views[0].same_as(views[1])


def test_frontend_shared_storage_rejects_incompatible_operand_layouts():
    with tvm.target.Target("ascend"), pytest.raises(tvm.error.InternalError, match="Get different layout"):
        transform.AscendLayoutInference()(frontend_module())


@pytest.mark.parametrize("keep_marker", [False, True], ids=["ordinary-aliases", "independent-scratch"])
def test_normalized_qk_pv_views_require_explicit_independent_layouts(keep_marker):
    @T.prim_func
    def main(
        Q: T.Tensor((64, 512), "bfloat16"),
        KV: T.Tensor((128, 512), "bfloat16"),
        P: T.Tensor((64, 128), "bfloat16"),
    ):
        with T.Kernel(1):
            q_l1 = T.alloc_l1((64, 512), "bfloat16")
            kv_l1 = T.alloc_l1((128, 512), "bfloat16")
            p_l1 = T.alloc_l1((64, 128), "bfloat16")
            accum = T.alloc_l0c((64, 512), "float32")
            T.copy(Q, q_l1)
            T.copy(KV, kv_l1)
            T.gemm(q_l1, kv_l1, accum[:, :128], transpose_B=True, clear_accum=True)
            T.copy(P, p_l1)
            T.gemm(p_l1, kv_l1, accum, transpose_B=False, clear_accum=True)

    with tvm.target.Target("ascend"):
        normalized = transform.NormalizeGemm()(frontend_module(main))
        declared = [buffer for block in nodes(normalized, tirx.SBlock) for buffer in block.annotations.get(INDEPENDENT_LAYOUT_BUFFERS, [])]
        operands = [gemm.args[index].args[0].buffer for gemm in calls(normalized, "tl.tileop.gemm") for index in (0, 1)]
        assert len(declared) == len(operands) == 4
        assert set(declared) == set(operands)

        if not keep_marker:

            def remove_marker(block):
                if isinstance(block, tirx.SBlock) and INDEPENDENT_LAYOUT_BUFFERS in block.annotations:
                    return tirx.SBlock(
                        block.iter_vars,
                        block.reads,
                        block.writes,
                        block.name_hint,
                        block.body,
                        block.init,
                        block.alloc_buffers,
                        block.match_buffers,
                        {key: value for key, value in block.annotations.items() if key != INDEPENDENT_LAYOUT_BUFFERS},
                    )

            normalized["main"] = normalized["main"].with_body(ir_transform(normalized["main"].body, None, remove_marker))
            with pytest.raises(tvm.error.InternalError, match="Get different layout"):
                transform.AscendLayoutInference()(normalized)
            return

        inferred = transform.AscendLayoutInference()(normalized)

    assert all(INDEPENDENT_LAYOUT_BUFFERS not in block.annotations for block in nodes(inferred, tirx.SBlock))
    layouts = _layouts(inferred)
    for scope in ("shared.l0a", "shared.l0b"):
        views = [buffer for buffer in operands if buffer.scope() == scope]
        assert [tuple(int(dim) for dim in view.shape) for view in views] == [(128, 128), (512, 32)]
        _assert_common_allocation(inferred, views)
        for view in views:
            assert layouts[view].is_equal(make_ascend_major_k_layout(view))


def test_marked_same_shape_aliases_still_require_identical_layouts():
    @T.prim_func
    def main():
        with T.Kernel(1):
            source = T.alloc_l1((32, 32), "bfloat16")
            a = T.alloc_l0a((32, 32), "bfloat16")
            b = T.alloc_l0b((32, 32), "bfloat16")
            alias = T.view(b, (32, 32))
            accum = T.alloc_l0c((32, 32), "float32")
            sblock_attr({INDEPENDENT_LAYOUT_BUFFERS: [b, alias]})
            T.copy(source, b)
            T.gemm(a, b, accum, transpose_B=True, clear_accum=True)
            T.copy(source, alias)
            T.gemm(a, alias, accum, transpose_B=False, clear_accum=True)

    with tvm.target.Target("ascend"), pytest.raises(tvm.error.InternalError, match="Get different layout"):
        transform.AscendLayoutInference()(frontend_module(main))


def test_read_only_cube_alias_keeps_its_producers_address_mapping():
    @T.prim_func
    def main(QK: T.Tensor((128, 128), "bfloat16")):
        with T.Kernel(1):
            base = T.alloc_l1((128, 128), "bfloat16")
            alias = T.view(base, (512, 32))
            dst = T.alloc_l0b((512, 32), "bfloat16")
            T.copy(QK, base)
            T.copy(alias, dst)

    target = tvm.target.Target("ascend")
    mod = tvm.IRModule({"main": main.with_attr("target", target)})
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    with target:
        inferred = transform.AscendLayoutInference()(mod)
    write, read = calls(inferred, "tl.tileop.ascend_copy")
    base = write.args[1].args[0].buffer
    alias = read.args[0].args[0].buffer
    layouts = _layouts(inferred)
    assert base.data.same_as(alias.data)
    assert tuple(int(dim) for dim in alias.shape) == (512, 32)
    assert layouts[alias].is_equal(layouts[base].reshape(alias.shape))


def test_unanchored_read_with_multiple_cube_mappings_is_rejected():
    with tvm.target.Target("ascend"), pytest.raises(tvm.error.InternalError, match="Get different layout"):
        transform.AscendLayoutInference()(frontend_module(shared_l0b_program(unanchored_read=True)))


def test_layout_reshape_rejects_different_capacity():
    layout = Layout((16, 32), lambda i, j: i * 32 + j)
    with pytest.raises(tvm.error.InternalError, match=r"InputShape\(\)"):
        layout.reshape((16, 64))


def test_layout_reshape_reinterpret_preserves_total_storage_bits():
    layout = Layout((16, 32), lambda i, j: i * 32 + j)
    reinterpreted = layout.reshape((16, 64), rescale_num=16, rescale_den=8)
    assert reinterpreted.is_equal(Layout((16, 64), lambda i, j: i * 64 + j))


def test_normalization_preserves_independent_view_addresses():
    normalized = transform.NormalizeAscendFractalStorage()(canonical_view_module())
    stores = nodes(normalized, tirx.BufferStore)
    views = [store.buffer for store in stores]
    _assert_common_allocation(normalized, views)
    layouts = _layouts(normalized)
    offsets = []
    for store in stores:
        assert tuple(int(index) for index in store.indices) == (16, 16)
        layout = layouts[store.buffer]
        physical = layout.map_forward_index(list(store.indices))
        offset = 0
        for coordinate, extent in zip(physical, layout.get_output_shape()):
            offset = offset * int(extent) + int(coordinate)
        offsets.append(offset)
    assert offsets == [2304, 8448]
    for pointer, view in zip(calls(normalized, "tl.access_ptr"), views):
        assert pointer.args[0].buffer.same_as(view)
        assert tuple(int(index) for index in pointer.args[0].indices) == (16, 16)
    tvm.ir.assert_structural_equal(normalized, transform.NormalizeAscendFractalStorage()(normalized))


def test_independent_views_reject_different_storage_capacities():
    with pytest.raises(tvm.error.InternalError, match="Independent Cube views must have equal padded storage capacity"):
        transform.NormalizeAscendFractalStorage()(canonical_view_module((1024, 32)))


if __name__ == "__main__":
    tilelang.testing.main()
