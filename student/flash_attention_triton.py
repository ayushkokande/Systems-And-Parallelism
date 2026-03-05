import math

import torch
import triton
import triton.language as tl
from torch.autograd import Function

from student.flash_attention import _flash_attention_backward


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    q_tile_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_idx * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(q_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_idx * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_idx * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_idx * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(q_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_idx * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(q_tile_idx * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Q_tile = tl.load(Q_block_ptr)

    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_tile = tl.load(K_block_ptr)
        V_tile = tl.load(V_block_ptr)

        S_ij = tl.dot(Q_tile, tl.trans(K_tile)) * scale

        if is_causal:
            q_offs = q_tile_idx * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_offs = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            S_ij = tl.where(q_offs[:, None] >= k_offs[None, :], S_ij, -1e6)

        m_ij = tl.max(S_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        P_ij = tl.exp(S_ij - m_new[:, None])
        l_new = tl.exp(m_i - m_new) * l_i + tl.sum(P_ij, axis=1)

        O_i = O_i * tl.exp(m_i - m_new)[:, None]
        O_i += tl.dot(P_ij.to(V_tile.dtype), V_tile)

        m_i = m_new
        l_i = l_new

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_i = O_i / l_i[:, None]
    L_i = m_i + tl.log(l_i)

    tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, L_i.to(L_block_ptr.type.element_ty))


class TritonFlashAttentionFunction(Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        batch, n_queries, d = q.shape
        n_keys = k.shape[1]

        O = torch.empty_like(q)
        L = torch.empty(batch, n_queries, device=q.device, dtype=q.dtype)

        Q_TILE = min(64, n_queries)
        K_TILE = min(64, n_keys)
        grid = (triton.cdiv(n_queries, Q_TILE), batch)

        flash_fwd_kernel[grid](
            q, k, v, O, L,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            n_queries, n_keys,
            1.0 / math.sqrt(d),
            D=d, Q_TILE_SIZE=Q_TILE, K_TILE_SIZE=K_TILE,
            is_causal=is_causal,
        )

        ctx.save_for_backward(q, k, v, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, do):
        q, k, v, O, L = ctx.saved_tensors
        dq, dk, dv = _flash_attention_backward(q, k, v, O, L, do, ctx.is_causal)
        return dq, dk, dv, None
