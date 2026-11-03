"""Collect, for every token vLLM is about to run, that token and the ones before it.

Pure torch on purpose: no vLLM imports, so it can be tested on CPU. All sizes come in
as Python ints and nothing reads GPU values back to the CPU, so it never stalls the
GPU and is safe to run before every step (including CUDA-graph replays).
"""
import torch


def fill_engram_ids(
    out: torch.Tensor,            # [max_num_tokens, max_order] int32, written in place
    input_ids: torch.Tensor,      # [>= num_tokens] this step's tokens, requests back to back
    positions: torch.Tensor,      # [>= num_tokens] position of each token in its request
    idx_mapping: torch.Tensor,    # [>= num_reqs] batch row -> row in all_token_ids
    query_start_loc: torch.Tensor,  # [>= num_reqs + 1] first token of each request
    all_token_ids: torch.Tensor,  # [max_num_reqs, max_model_len] every request's tokens so far
    num_reqs: int,
    num_tokens: int,
    num_tokens_padded: int,
) -> None:
    """out[t] = (token at t, token at t-1, token at t-2, ...), -1 where there is none.

    Earlier tokens come from this step's input_ids when they are in the same chunk,
    otherwise from the request's history (earlier chunks, earlier decode steps,
    prefix-cache hits, or tokens restored after preemption).
    """
    max_order = out.shape[1]
    if num_tokens > 0:
        dev = input_ids.device
        qsl = query_start_loc[: num_reqs + 1].long()
        begin = qsl[:-1]
        req_of_token = torch.repeat_interleave(
            torch.arange(num_reqs, device=dev), qsl[1:] - begin, output_size=num_tokens
        )
        pos = positions[:num_tokens].long()
        ids = input_ids[:num_tokens].long()
        chunk_start = pos[begin.clamp(max=num_tokens - 1)]  # first position run this step
        state = idx_mapping[:num_reqs].long()

        # The (max_order - 1) tokens right before each request's chunk, from its history.
        lookback = []
        for j in range(max_order - 1):
            q = chunk_start - 1 - j
            hist = all_token_ids[state, q.clamp(min=0)].long()
            lookback.append(torch.where(q >= 0, hist, torch.full_like(hist, -1)))

        out[:num_tokens, 0] = ids
        if max_order > 1:
            lookback = torch.stack(lookback, dim=1)  # [num_reqs, max_order - 1]
            tok = torch.arange(num_tokens, device=dev)
            start = chunk_start[req_of_token]
            for k in range(1, max_order):
                q = pos - k
                from_chunk = ids[(tok - k).clamp(min=0)]
                j = (start - 1 - q).clamp(0, max_order - 2)
                from_hist = lookback[req_of_token, j]
                out[:num_tokens, k] = torch.where(q >= start, from_chunk, from_hist)
    if num_tokens_padded > num_tokens:
        out[num_tokens:num_tokens_padded] = -1  # CUDA-graph padding rows
