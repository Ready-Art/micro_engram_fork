"""Model Runner V2 state that feeds the engram its token ids before every step."""
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from .ids import fill_engram_ids


class EngramModelState(DefaultModelState):
    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        # Owned by the model, so its address never changes (CUDA graphs read it in place).
        self.engram_ids = getattr(model, "micro_engram_ids", None)
        if self.engram_ids is not None and self.engram_ids.shape[0] < self.max_num_tokens:
            raise ValueError(
                f"[micro_engram] token-id buffer holds {self.engram_ids.shape[0]} tokens, "
                f"the runner schedules up to {self.max_num_tokens}"
            )

    def prepare_inputs(self, input_batch, req_states):
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if self.engram_ids is not None:
            fill_engram_ids(
                self.engram_ids,
                input_batch.input_ids,
                input_batch.positions,
                input_batch.idx_mapping,
                input_batch.query_start_loc,
                req_states.all_token_ids.gpu,
                num_reqs=input_batch.num_reqs,
                num_tokens=input_batch.num_tokens,
                num_tokens_padded=input_batch.num_tokens_after_padding,
            )
        return model_inputs
