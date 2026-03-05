import math

import torch
from torch.autograd import Function
from einops import einsum


def _flash_attention_forward(q, k, v, is_causal, block_size=64):
    *batch_dims, n_queries, d = q.shape
    n_keys = k.shape[-2]
    scale = 1.0 / math.sqrt(d)

    q_flat = q.reshape(-1, n_queries, d)
    k_flat = k.reshape(-1, n_keys, d)
    v_flat = v.reshape(-1, n_keys, d)

    Br = min(block_size, n_queries)
    Bc = min(block_size, n_keys)
    batch = q_flat.shape[0]

    O = torch.zeros(batch, n_queries, d, device=q.device, dtype=q.dtype)
    L = torch.full((batch, n_queries), float("-inf"), device=q.device, dtype=q.dtype)

    for i in range(0, n_queries, Br):
        i_end = min(i + Br, n_queries)
        q_i = q_flat[:, i:i_end, :]

        m_i = torch.full((batch, i_end - i), float("-inf"), device=q.device, dtype=q.dtype)
        ell_i = torch.zeros(batch, i_end - i, device=q.device, dtype=q.dtype)
        O_i = torch.zeros(batch, i_end - i, d, device=q.device, dtype=q.dtype)

        for j in range(0, n_keys, Bc):
            j_end = min(j + Bc, n_keys)
            k_j = k_flat[:, j:j_end, :]
            v_j = v_flat[:, j:j_end, :]

            S_ij = einsum(q_i, k_j, "b q d, b k d -> b q k") * scale

            if is_causal:
                q_idx = torch.arange(i, i_end, device=q.device).view(1, -1, 1)
                k_idx = torch.arange(j, j_end, device=q.device).view(1, 1, -1)
                causal_mask = q_idx >= k_idx
                S_ij = torch.where(causal_mask, S_ij, float("-inf"))

            m_ij = S_ij.max(dim=-1, keepdim=True).values.squeeze(-1)  
            m_new = torch.maximum(m_i, m_ij)

            P_ij = torch.exp(S_ij - m_new.unsqueeze(-1))
            ell_new = torch.exp(m_i - m_new) * ell_i + P_ij.sum(dim=-1)

            O_i = torch.exp(m_i.unsqueeze(-1) - m_new.unsqueeze(-1)) * O_i
            O_i = O_i + einsum(P_ij, v_j, "b q k, b k d -> b q d")

            m_i = m_new
            ell_i = ell_new

        O[:, i:i_end, :] = O_i / ell_i.unsqueeze(-1).clamp(min=1e-6)
        L[:, i:i_end] = m_i + torch.log(ell_i.clamp(min=1e-6))

    O = O.reshape(*batch_dims, n_queries, d)
    L = L.reshape(*batch_dims, n_queries)

    return O, L


def _flash_attention_backward(q, k, v, O, L, do, is_causal, block_size=64):
    *batch_dims, n_queries, d = q.shape
    n_keys = k.shape[-2]
    scale = 1.0 / math.sqrt(d)

    q_flat = q.reshape(-1, n_queries, d)
    k_flat = k.reshape(-1, n_keys, d)
    v_flat = v.reshape(-1, n_keys, d)
    O_flat = O.reshape(-1, n_queries, d)
    do_flat = do.reshape(-1, n_queries, d)
    L_flat = L.reshape(-1, n_queries)

    D = (O_flat * do_flat).sum(dim=-1)

    batch = q_flat.shape[0]
    Br = min(block_size, n_queries)
    Bc = min(block_size, n_keys)

    dq = torch.zeros_like(q_flat)
    dk = torch.zeros_like(k_flat)
    dv = torch.zeros_like(v_flat)

    for i in range(0, n_queries, Br):
        i_end = min(i + Br, n_queries)
        q_i = q_flat[:, i:i_end, :]
        do_i = do_flat[:, i:i_end, :]
        L_i = L_flat[:, i:i_end]
        D_i = D[:, i:i_end]

        for j in range(0, n_keys, Bc):
            j_end = min(j + Bc, n_keys)
            k_j = k_flat[:, j:j_end, :]
            v_j = v_flat[:, j:j_end, :]

            S_ij = einsum(q_i, k_j, "b q d, b k d -> b q k") * scale

            if is_causal:
                q_idx = torch.arange(i, i_end, device=q.device).view(1, -1, 1)
                k_idx = torch.arange(j, j_end, device=q.device).view(1, 1, -1)
                causal_mask = q_idx >= k_idx
                S_ij = torch.where(causal_mask, S_ij, float("-inf"))

            P_ij = torch.exp(S_ij - L_i.unsqueeze(-1))

            dv[:, j:j_end, :] += einsum(P_ij, do_i, "b q k, b q d -> b k d")
            dP_ij = einsum(do_i, v_j, "b q d, b k d -> b q k")
            dS_ij = P_ij * (dP_ij - D_i.unsqueeze(-1))

            dq[:, i:i_end, :] += einsum(dS_ij, k_j, "b q k, b k d -> b q d") * scale
            dk[:, j:j_end, :] += einsum(dS_ij, q_i, "b q k, b q d -> b k d") * scale

    dq = dq.reshape(*batch_dims, n_queries, d)
    dk = dk.reshape(*batch_dims, n_keys, d)
    dv = dv.reshape(*batch_dims, n_keys, d)

    return dq, dk, dv


class FlashAttentionFunction(Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal, block_size=64):
        O, L = _flash_attention_forward(q, k, v, is_causal, block_size)
        ctx.save_for_backward(q, k, v, O, L)
        ctx.is_causal = is_causal
        ctx.block_size = block_size
        return O

    @staticmethod
    def backward(ctx, do):
        q, k, v, O, L = ctx.saved_tensors
        dq, dk, dv = _flash_attention_backward(q, k, v, O, L, do, ctx.is_causal, ctx.block_size)
        return dq, dk, dv, None, None
