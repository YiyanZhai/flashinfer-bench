import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from typing_extensions import override

from flashinfer_bench.bench.config import BenchmarkConfig
from flashinfer_bench.bench.evaluators.default import DefaultEvaluator
from flashinfer_bench.bench.runner.runner import BaselineHandle, DeviceBaseline
from flashinfer_bench.bench.timing import time_runnable
from flashinfer_bench.bench.utils import (
    compute_error_stats,
    gen_inputs,
    load_safetensors,
    make_eval,
    normalize_outputs,
)
from flashinfer_bench.compile.registry import get_builder_registry
from flashinfer_bench.compile.runnable import Runnable
from flashinfer_bench.data.definition import Definition
from flashinfer_bench.data.trace import Correctness, Evaluation, EvaluationStatus, Workload
from flashinfer_bench.utils import dtype_str_to_torch_dtype


class SamplingEvaluator(DefaultEvaluator):

    @override
    @classmethod
    def can_evaluate(cls, defn: Definition) -> bool:
        return is_sampling_op(defn)

    @override
    @classmethod
    def build_baseline(
        cls,
        defn: Definition,
        workload: Workload,
        cfg: BenchmarkConfig,
        device: str,
        traceset_root: Optional[Path] = None,
    ) -> DeviceBaseline:
        ref_runnable = get_builder_registry().build_reference(defn)
        loaded_stensors = (
            load_safetensors(defn, workload, traceset_root)
            if any(d.type == "safetensors" for d in workload.inputs.values())
            else {}
        )

        inputs: List[Dict[str, Any]] = []
        outputs: List[Dict[str, torch.Tensor]] = []

        inp = gen_inputs(defn, workload, device=device, stensors=loaded_stensors)
        if "probs" in inp:
            inp["probs"] = torch.softmax(
                inp["probs"], dim=-1
            )  # convert logits to probs for sampling
        inputs.append(inp)

        thresholding_method = _detect_thresholding_method(defn)
        params = {k: inp[k] for k in ["top_k", "top_p"] if k in inp}
        valid_mask = _compute_valid_sampling_mask(inp["probs"], thresholding_method, params)
        
        masked_probs = inp["probs"] * valid_mask.float()
        expected_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
        
        outputs.append({"expected_probs": expected_probs})

        latencies: List[float] = []
        for inp in inputs:
            ms = time_runnable(ref_runnable, inp, cfg.warmup_runs, cfg.iterations, device)
            latencies.append(ms)

        mean_latency_ms = sum(latencies) / float(len(latencies))

        handle = BaselineHandle(uuid.uuid4().hex)

        return DeviceBaseline(
            handle=handle,
            defn=defn,
            device=device,
            inputs=inputs,
            outputs=outputs,
            mean_latency_ms=mean_latency_ms,
        )

    @override
    @classmethod
    def check_correctness(
        cls,
        defn: Definition,
        sol_runnable: Runnable,
        inputs: List[Dict[str, Any]],
        ref_outputs: List[Dict[str, torch.Tensor]],
        cfg: BenchmarkConfig,
        log_path: str,
        device: str,
    ) -> Tuple[Optional[Correctness], Optional[Evaluation]]:
        expected_probs = ref_outputs[0]["expected_probs"]
        vocab_size = expected_probs.shape[-1]

        inp = inputs[0]
        params = {k: inp[k] for k in ["top_k", "top_p"] if k in inp}

        output_names = list(defn.outputs.keys())
        output_dtypes = {k: dtype_str_to_torch_dtype(v.dtype) for k, v in defn.outputs.items()}

        # Compute valid sampling mask based on thresholding
        thresholding_method = _detect_thresholding_method(defn)
        probs = inp["probs"]
        valid_mask = _compute_valid_sampling_mask(probs, thresholding_method, params)

        # Validate correct sampling token set
        for _ in range(cfg.sampling_validation_trials):
            try:
                with torch.no_grad():
                    out = sol_runnable(**inp)
                torch.cuda.synchronize(device)
            except Exception:
                traceback.print_exc()
                return None, make_eval(
                    status=EvaluationStatus.RUNTIME_ERROR, device=device, log_path=log_path
                )

            out_normalized = normalize_outputs(
                out, device=device, output_names=output_names, output_dtypes=output_dtypes
            )
            samples = out_normalized["samples"]

            # Check vocabulary range
            if (samples < 0).any() or (samples >= vocab_size).any():
                invalid_samples = samples[(samples < 0) | (samples >= vocab_size)]
                correctness = Correctness(
                    max_relative_error=float("inf"), max_absolute_error=float("inf")
                )
                message = (
                    f"Samples {invalid_samples.tolist()} out of vocabulary range [0, {vocab_size})"
                )
                print(message, file=sys.stderr)
                return correctness, make_eval(
                    status=EvaluationStatus.INCORRECT_NUMERICAL,
                    device=device,
                    log_path=log_path,
                    correctness=correctness,
                )

            # Validate thresholding - check samples are within valid mask
            if samples.dim() == 0:
                samples_flat = samples.unsqueeze(0)
            else:
                samples_flat = samples.flatten()
            
            batch_size = valid_mask.shape[0]
            for i in range(len(samples_flat)):
                batch_idx = i % batch_size
                sample_idx = samples_flat[i].item()
                if not valid_mask[batch_idx, sample_idx]:
                    correctness = Correctness(
                        max_relative_error=float("inf"), max_absolute_error=float("inf")
                    )
                    message = (
                        f"Sample {sample_idx} is outside valid {thresholding_method} mask for batch {batch_idx}"
                    )
                    print(message, file=sys.stderr)
                    return correctness, make_eval(
                        status=EvaluationStatus.INCORRECT_NUMERICAL,
                        device=device,
                        log_path=log_path,
                        correctness=correctness,
                    )

        try:
            sol_freq = _compute_frequency_distribution(
                sol_runnable, inp, device, defn, num_trials=500000
            )
            torch.cuda.synchronize(device)
        except Exception:
            traceback.print_exc()
            return None, make_eval(
                status=EvaluationStatus.RUNTIME_ERROR, device=device, log_path=log_path
            )

        # Expected frequency is the masked and renormalized ground truth averaged across batch
        expected_freq = expected_probs.mean(dim=0)

        # total variation distance
        tvd = 0.5 * torch.sum(torch.abs(sol_freq - expected_freq)).item()
        max_abs, max_rel, _, _ = compute_error_stats(sol_freq, expected_freq, cfg)

        numerical_incorrect = tvd > cfg.sampling_tvd_threshold
        correctness = Correctness(
            max_relative_error=max_rel, max_absolute_error=max_abs, extra={"tvd": tvd}
        )
        if numerical_incorrect:
            return correctness, make_eval(
                status=EvaluationStatus.INCORRECT_NUMERICAL,
                device=device,
                log_path=log_path,
                correctness=correctness,
            )

        return correctness, None


def is_sampling_op(defn: Definition) -> bool:
    return getattr(defn, "op_type", None) == "sampling"


def _detect_thresholding_method(defn: Definition) -> str:
    name = defn.name.lower()
    if "top_k_top_p" in name:
        return "top_k_top_p"
    elif "top_k" in name:
        return "top_k"
    elif "top_p" in name:
        return "top_p"
    else:
        return "none"  # no thresholding


def _compute_valid_sampling_mask(
    probs: torch.Tensor, method: str, params: Dict[str, Any], eps: float = 1e-5
) -> torch.Tensor:
    """Compute valid sampling mask based on thresholding method.

    Handles tie-breaking in top_k (allows any token with prob >= k-th largest)
    and numerical precision in top_p (allows tokens within eps of nucleus boundary).

    Parameters
    ----------
    probs : torch.Tensor
        Probability distribution, shape [batch_size, vocab_size].
    method : str
        Thresholding method: "top_k", "top_p", "top_k_top_p", or "none".
    params : Dict[str, Any]
        Sampling parameters (top_k, top_p values).
    eps : float
        Epsilon tolerance for top_p nucleus boundary (default 1e-5).

    Returns
    -------
    torch.Tensor
        Boolean mask of shape [batch_size, vocab_size] indicating valid tokens.
    """
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)
    
    batch_size, vocab_size = probs.shape
    device = probs.device

    if method == "none":
        return torch.ones((batch_size, vocab_size), dtype=torch.bool, device=device)

    mask = torch.ones((batch_size, vocab_size), dtype=torch.bool, device=device)

    # Apply top_k mask with tie-breaking support
    if method in ["top_k", "top_k_top_p"]:
        if "top_k" not in params:
            raise ValueError(f"top_k parameter required for {method} but not found")
        
        top_k_param = params["top_k"]
        for i in range(batch_size):
            k = (
                int(top_k_param[i].item())
                if top_k_param.dim() > 0
                else int(top_k_param.item())
            )

            if 0 < k < vocab_size:
                # Sort probabilities in descending order
                sorted_probs, _ = torch.sort(probs[i], descending=True)
                # k-th largest value (0-indexed, so k-1)
                pivot = sorted_probs[k - 1]
                # Allow any token with prob >= pivot (handles ties)
                mask[i] = probs[i] >= pivot
            # If k <= 0 or k >= vocab_size, keep all tokens valid

    # Apply top_p mask with epsilon tolerance
    if method in ["top_p", "top_k_top_p"]:
        if "top_p" not in params:
            raise ValueError(f"top_p parameter required for {method} but not found")
        
        top_p_param = params["top_p"]
        for i in range(batch_size):
            p = (
                float(top_p_param[i].item())
                if top_p_param.dim() > 0
                else float(top_p_param.item())
            )

            if 0 < p < 1:
                # Sort in descending order for cumulative sum
                sorted_probs, sorted_indices = torch.sort(probs[i], descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=0)
                
                # Find tokens in nucleus (cumsum <= p + eps for numerical tolerance)
                nucleus_mask = cumsum <= (p + eps)
                
                # Always include at least one token (the highest probability one)
                if not nucleus_mask.any():
                    nucleus_mask[0] = True
                
                # Map back to original indices
                p_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
                p_mask[sorted_indices[nucleus_mask]] = True
                
                # Combine with existing mask (intersection)
                mask[i] = mask[i] & p_mask
            # If p <= 0 or p >= 1, keep all tokens valid (no filtering)

    return mask


def _compute_frequency_distribution(
    runnable: Runnable,
    inputs: Dict[str, Any],
    device: str,
    defn: Definition,
    num_trials: int = 500000,
) -> torch.Tensor:
    original_batch_size = inputs["probs"].shape[0] if inputs["probs"].dim() > 1 else 1
    vocab_size = inputs["probs"].shape[-1]
    
    # Pad inputs to batch size 10,000 for more efficient sampling
    target_batch_size = 10000
    padded_inputs = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor) and value.dim() > 0:
            if key == "probs":
                # For probs, repeat the first batch element to fill target_batch_size
                if value.dim() == 1:
                    value = value.unsqueeze(0)
                # Repeat the first element to fill target_batch_size
                repeat_count = target_batch_size
                padded_value = value[0:1].repeat(repeat_count, *([1] * (value.dim() - 1)))
            elif key in ["top_k", "top_p"]:
                # For sampling parameters, repeat the first value
                if value.dim() == 0:
                    padded_value = value.unsqueeze(0).repeat(target_batch_size)
                else:
                    padded_value = value[0:1].repeat(target_batch_size)
            else:
                # For other tensors, repeat along batch dimension
                if value.dim() == 0:
                    padded_value = value.unsqueeze(0).repeat(target_batch_size)
                else:
                    padded_value = value[0:1].repeat(target_batch_size, *([1] * (value.dim() - 1)))
            padded_inputs[key] = padded_value
        else:
            # For non-tensor inputs, keep as is
            padded_inputs[key] = value
    
    batch_size = target_batch_size
    counter = torch.zeros(vocab_size, dtype=torch.int64, device=torch.device(device))

    trials_needed = (num_trials + batch_size - 1) // batch_size
    total_samples_collected = 0

    for _ in range(trials_needed):
        with torch.no_grad():
            out = runnable(**padded_inputs)

        output_names = list(defn.outputs.keys())
        output_dtypes = {k: dtype_str_to_torch_dtype(v.dtype) for k, v in defn.outputs.items()}

        out_normalized = normalize_outputs(
            out, device=torch.device(device), output_names=output_names, output_dtypes=output_dtypes
        )

        samples = out_normalized["samples"]

        if samples.dim() == 0:
            sample_idx = samples.item()
            counter[sample_idx] += 1
            total_samples_collected += 1
        else:  # Batch of samples
            for i in range(samples.numel()):
                sample_idx = samples.flatten()[i].item()
                counter[sample_idx] += 1
                total_samples_collected += 1

    frequency = counter.float() / total_samples_collected
    return frequency
