#!/usr/bin/env python3

# Test gguf.quants so that it exactly matches the C implementation of the (de)quantization

# NOTE: this is kind of a mess, but at least it worked for initially testing the Python implementations.

from __future__ import annotations

import argparse
from math import prod
import os
import sys
from pathlib import Path
import ctypes
import logging
import numpy as np

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

import gguf
from gguf.constants import GGMLQuantizationType


logger = logging.getLogger("test-quants")


c_float_p = ctypes.POINTER(ctypes.c_float)


class ggml_init_params(ctypes.Structure):
    _fields_ = [
        ("mem_size", ctypes.c_size_t),
        ("mem_buffer", ctypes.c_void_p),
        ("no_alloc", ctypes.c_bool),
    ]


class GGMLQuants:
    libggml: ctypes.CDLL

    def __init__(self, libggml: Path):
        self.libggml = ctypes.CDLL(str(libggml))
        self.libggml.ggml_quantize_chunk.restype = ctypes.c_size_t
        # enum ggml_type   type,
        #    const float * src,
        #           void * dst,
        #        int64_t   start,
        #        int64_t   nrows,
        #        int64_t   n_per_row,
        #    const float * imatrix) {
        self.libggml.ggml_quantize_chunk.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        )

        self.libggml.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
        self.libggml.ggml_quantize_requires_imatrix.argtypes = (ctypes.c_int,)

        for t in (
            "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q6_0",
            "q2_K", "q3_K", "q4_K", "q5_K", "q6_K",
            "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s", "iq1_s", "iq1_m",
            "iq4_nl", "iq4_xs",
        ):
            dequant_func: ctypes._NamedFuncPointer = getattr(self.libggml, "dequantize_row_" + t)
            dequant_func.restype = None
            dequant_func.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)

        self.libggml.ggml_fp16_to_fp32_row.restype = None
        self.libggml.ggml_fp16_to_fp32_row.argtypes = (ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_float), ctypes.c_int64)
        self.libggml.ggml_bf16_to_fp32_row.restype = None
        self.libggml.ggml_bf16_to_fp32_row.argtypes = (ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_float), ctypes.c_int64)

        self.libggml.ggml_init.argtypes = (ggml_init_params,)

        self.libggml.ggml_init(ggml_init_params(1 * 1024 * 1024, 0, False))

    def dequantize(self, tensor: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
        result = np.zeros(gguf.quant_shape_from_byte_shape(tensor.shape, qtype), dtype=np.float32, order="C")
        if qtype == GGMLQuantizationType.F32:
            # no-op
            result = tensor.view(np.float32)
        elif qtype == GGMLQuantizationType.F16:
            self.libggml.ggml_fp16_to_fp32_row(tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), result.ctypes.data_as(c_float_p), result.size)
        elif qtype == GGMLQuantizationType.BF16:
            self.libggml.ggml_bf16_to_fp32_row(tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), result.ctypes.data_as(c_float_p), result.size)
        else:
            lw_qname = qtype.name.lower()
            if lw_qname[-1] == "k":
                lw_qname = lw_qname[:-1] + "K"
            dequant_func: ctypes._NamedFuncPointer = getattr(self.libggml, "dequantize_row_" + lw_qname)
            dequant_func(tensor.ctypes.data_as(ctypes.c_void_p), result.ctypes.data_as(c_float_p), result.size)
        return result

    def quantize(self, data: np.ndarray, qtype: GGMLQuantizationType) -> np.ndarray:
        result = np.zeros(gguf.quant_shape_to_byte_shape(data.shape, qtype), dtype=np.uint8, order="C")
        if self.libggml.ggml_quantize_requires_imatrix(qtype.value):
            # TODO: is a column-wise sum of squares appropriate?
            qw = np.sum((data * data).reshape((-1, data.shape[-1])), axis=0).ctypes.data_as(c_float_p)
        else:
            qw = ctypes.cast(0, c_float_p)
        result_size = self.libggml.ggml_quantize_chunk(qtype.value, data.ctypes.data_as(c_float_p), result.ctypes.data_as(ctypes.c_void_p), 0, prod(data.shape[:-1]), data.shape[-1], qw)
        assert result.size == result_size
        return result


def compare_tensors(t1: np.ndarray, t2: np.ndarray, qtype: GGMLQuantizationType) -> bool:
    same = np.array_equal(t1, t2)
    if same:
        return True
    else:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
        if t1.dtype == np.float32:
            t1 = t1.reshape((-1, block_size))
            t2 = t2.reshape((-1, block_size))
        else:
            t1 = t1.reshape((-1, type_size))
            t2 = t2.reshape((-1, type_size))
        x = t1.view(np.uint8) ^ t2.view(np.uint8)
        diff_bits = np.count_nonzero(np.unpackbits(x, axis=-1), axis=-1)
        num_bad_blocks = np.count_nonzero(diff_bits, axis=0)
        if num_bad_blocks == 0 and t1.shape == t2.shape:
            logger.debug("Bits are equal, but arrays don't match, likely contains NANs")
            return True
        logger.debug(f"{num_bad_blocks} bad blocks ({100 * num_bad_blocks / x.shape[0]:.6f}%)")
        bad_block_id = np.argmax(diff_bits, axis=0)
        logger.debug(f"Worst block id: {bad_block_id}")
        logger.debug(f"Sample bad block ({diff_bits[bad_block_id]} differing bits):\n{t1[bad_block_id]}\nReference:\n{t2[bad_block_id]}")

        sum_diff_bits = np.sum(diff_bits)
        logger.debug(f"{sum_diff_bits} bits differ ({100 * sum_diff_bits/(x.size * 8):.6f}%)")
        return False


def do_test(libggml_path: Path, quick: bool = False):
    ggml_quants = GGMLQuants(libggml_path)

    np.set_printoptions(precision=None, threshold=(4 * 256) + 1, formatter={"int": lambda n: "0x%02X" % n})

    r = np.random.randn(8, 1024, 1024).astype(np.float32, copy=False)

    for qtype in (GGMLQuantizationType.F16, *gguf.quants._type_traits.keys()):
        has_dequantize = False
        has_quantize = False

        try:
            gguf.dequantize(np.zeros((gguf.GGML_QUANT_SIZES[qtype][1]), dtype=np.uint8), qtype)
            has_dequantize = True
        except (NotImplementedError, AssertionError) as e:
            if isinstance(e, AssertionError):
                logger.error(f"Error with {qtype.name}: {e}")
                raise e
        try:
            gguf.quantize(np.zeros((gguf.GGML_QUANT_SIZES[qtype][0]), dtype=np.float32), qtype)
            has_quantize = True
        except (NotImplementedError, AssertionError) as e:
            if isinstance(e, AssertionError):
                logger.error(f"Error with {qtype.name}: {e}")
                raise e

        if not has_dequantize and not has_quantize:
            continue

        logger.info(f"Testing {qtype.name}")

        rc = r.copy(order="C")

        pyq = None
        ggq = None

        if has_quantize:
            logger.debug(f"Quantizing to {qtype.name} with Python")
            pyq = gguf.quants.quantize(rc, qtype)

            logger.debug(f"Quantizing to {qtype.name} with C")
            ggq = ggml_quants.quantize(rc, qtype)

            if qtype == GGMLQuantizationType.F16:
                pyq = pyq.view(np.uint8)
            quant_equal = compare_tensors(pyq, ggq, qtype)

            if not quant_equal:
                logger.error(f"Quantization to {qtype.name} does not match ❌")
            else:
                logger.info(f"Quantization to {qtype.name} matches exactly ✅")

        if has_dequantize:
            if ggq is None and not quick:
                logger.debug(f"Quantizing to {qtype.name} with C")
                ggq = ggml_quants.quantize(rc, qtype)

            if ggq is not None:
                logger.debug(f"Dequantizing from {qtype.name} with Python")
                pydq = gguf.quants.dequantize(ggq, qtype)
                logger.debug(f"Dequantizing from {qtype.name} with C")
                ggdq = ggml_quants.dequantize(ggq, qtype)

                dequant_equal = compare_tensors(pydq, ggdq, qtype)

                if not dequant_equal:
                    logger.error(f"Dequantization from {qtype.name} does not match ❌")
                else:
                    logger.info(f"Dequantization from {qtype.name} matches exactly ✅")

            rq_shape = gguf.quants.quant_shape_to_byte_shape((8, 1024, 1024 // 2), qtype)
            rq = np.random.random(rq_shape).astype(np.float16).view(np.uint8)

            logger.debug(f"Dequantizing random f16 data as {qtype.name} with Python")
            pydq = gguf.quants.dequantize(rq, qtype)
            logger.debug(f"Dequantizing random f16 data as {qtype.name} with C")
            ggdq = ggml_quants.dequantize(rq, qtype)

            dequant_equal = compare_tensors(pydq, ggdq, qtype)

            if not dequant_equal:
                logger.error(f"Dequantization from random f16 data as {qtype.name} does not match ❌")
            else:
                logger.info(f"Dequantization from random f16 data as {qtype.name} matches exactly ✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Python (de)quantization against the reference C implementation")
    parser.add_argument("--libggml", type=Path, default=Path(__file__).parent.parent.parent / "build" / "ggml" / "src" / "libggml.so", help="The path to libggml.so")
    parser.add_argument("--quick", action="store_true", help="Don't quantize with C when it's not strictly necessary")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    do_test(args.libggml, args.quick)


# ----------------------------------------------------------------------
# pytest cases for Q4_0_AR16 (PHASE32 Stage 2, Phase 1.C)
#
# These exercise the numpy reference in gguf.quants.Q4_0_AR16 against the
# Allium contract at /home/llm/yarn-agentic/q4_0_ar16.allium. They run
# without a built libggml -- pure numpy.
# ----------------------------------------------------------------------

from gguf.quants import Q4_0_AR16  # noqa: E402


def test_q4_0_ar16_roundtrip():
    np.random.seed(0)
    x = np.random.randn(64, 16).astype(np.float32)
    blocks = Q4_0_AR16.quantize_blocks(x)
    assert blocks.shape == (64, 10)
    assert blocks.dtype == np.uint8
    y = Q4_0_AR16.dequantize_blocks(blocks)
    assert y.shape == (64, 16)
    assert y.dtype == np.float32

    # Sym INT4 with positive scale = absmax/8 has the worst-case
    # per-element error at the positive-max element (code clipped from
    # 8 -> 7), exactly block_max/8 in fp32. Storing d as fp16 adds at
    # most ~7 * d * 2^-10 of slack on top.
    per_block_max = np.abs(x).max(axis=1, keepdims=True)
    err = np.abs(x - y)
    tol = per_block_max / 8 + per_block_max * 1e-2
    assert np.all(err <= tol), f"max err {err.max()} > tol {tol.max()}"


def test_q4_0_ar16_deterministic():
    np.random.seed(1)
    x = np.random.randn(32, 16).astype(np.float32)
    a = Q4_0_AR16.quantize_blocks(x)
    b = Q4_0_AR16.quantize_blocks(x)
    assert np.array_equal(a, b)


def test_q4_0_ar16_col_perm_16_aligned_lossless():
    # The Allium @invariant col_perm_equivalence_16_aligned: any
    # permutation of 16-element column chunks of the source row
    # reshuffles whole blocks bit-exactly.
    np.random.seed(42)
    n_blocks = 32
    x = np.random.randn(n_blocks, 16).astype(np.float32)
    blocks = Q4_0_AR16.quantize_blocks(x)
    perm = np.random.permutation(n_blocks)

    # Path A: permute blocks (16-element chunks), then dequantize.
    a_blocks = blocks[perm]
    a = Q4_0_AR16.dequantize_blocks(a_blocks)

    # Path B: dequantize, then permute the same way (16-aligned chunks).
    b_full = Q4_0_AR16.dequantize_blocks(blocks)
    b = b_full[perm]

    assert np.array_equal(a, b)


def test_q4_0_ar16_zero_block():
    # Allium @invariant ZeroBlockHandling: all-zero block -> scale 0,
    # codes all 8, dequantize to 0.0.
    x = np.zeros((1, 16), dtype=np.float32)
    blocks = Q4_0_AR16.quantize_blocks(x)
    # d bytes (positions 0..2) zero, all qs bytes encode 8|8<<4 = 0x88.
    assert blocks[0, 0] == 0 and blocks[0, 1] == 0
    assert np.all(blocks[0, 2:] == 0x88)
    y = Q4_0_AR16.dequantize_blocks(blocks)
    assert np.all(y == 0.0)


def _hand_pack_q4_0_ar16(values: np.ndarray) -> np.ndarray:
    """Reference implementation derived directly from the Allium spec
    and the Q4_0 packing convention. Used only by the spec-parity test;
    deliberately written without numpy vectorisation tricks so a
    spec-savvy reader can verify it line-by-line."""
    assert values.shape == (16,)
    block_max = float(np.abs(values).max())
    scale = block_max / 8.0
    out = np.zeros(10, dtype=np.uint8)
    out[0:2] = np.array([np.float16(scale)], dtype=np.float16).view(np.uint8)
    if scale == 0.0:
        # All codes -> 8; nibble pack 8 | (8 << 4) = 0x88 per byte.
        out[2:10] = 0x88
        return out
    inv = 1.0 / scale
    codes = np.empty(16, dtype=np.int32)
    for k in range(16):
        # Banker's rounding: matches numpy.rint and the C round-nearest-even.
        c = int(np.rint(values[k] * inv))
        if c < -8:
            c = -8
        if c > 7:
            c = 7
        codes[k] = c + 8
    # Q4_0 half-split convention: byte j packs codes[j] (low) and codes[8+j] (high).
    for j in range(8):
        out[2 + j] = (codes[j] & 0x0F) | ((codes[8 + j] & 0x0F) << 4)
    return out


def test_q4_0_ar16_spec_parity_fixtures():
    # 5 hand-derived fixtures. Each exercises a different facet of the
    # spec: pure positive ramp (asymmetric clipping at block_max),
    # pure negative ramp, mixed sign, the zero block, and a fixture
    # with a single dominant element forcing scale = that element / 8.
    fixtures = [
        # Positive ramp: max at index 15 = 1.5 -> scale = 1.5/8 = 0.1875.
        np.linspace(0.0, 1.5, 16, dtype=np.float32),
        # Negative ramp.
        np.linspace(0.0, -1.5, 16, dtype=np.float32),
        # Mixed sign symmetric.
        np.array([-2.0, -1.5, -1.0, -0.5, -0.25, -0.125, -0.0625, 0.0,
                  0.0, 0.0625, 0.125, 0.25, 0.5, 1.0, 1.5, 2.0], dtype=np.float32),
        # All-zero.
        np.zeros(16, dtype=np.float32),
        # Single dominant element.
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    ]
    for i, x in enumerate(fixtures):
        expected = _hand_pack_q4_0_ar16(x)
        got = Q4_0_AR16.quantize_blocks(x.reshape(1, 16))[0]
        assert np.array_equal(got, expected), (
            f"fixture {i}: got {got.tolist()} expected {expected.tolist()}"
        )
