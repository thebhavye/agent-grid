"""Measure the real advisor on this host: python -m edge_llm.benchmark.

Set AGENTGRID_GGUF_MODEL or --model to measure GGUF. No downloads are made.
"""

import argparse
import json
import platform
import time

from edge_llm import FaultAdvisor


def peak_memory():
    """Return peak process MiB and an unavailability reason, without failing."""
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024 ** 2 if platform.system() == "Darwin" else 1024), None
    except (ImportError, OSError, AttributeError):
        pass
    try:
        import psutil
        info = psutil.Process().memory_info()
        return info.peak_wset / 1024 ** 2, None
    except Exception as exc:
        return None, ("Peak memory unavailable: resource unsupported and Windows "
                      f"peak_wset unavailable ({type(exc).__name__}).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    args = parser.parse_args()
    advisor = FaultAdvisor(args.model)
    event = {"fault_type": "inverter_trip", "device": "solar", "start_step": 144,
             "end_step": 156, "coincident_unmet_kwh": 0.5, "readings": []}
    started = time.perf_counter()
    advisor._load()
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    text = advisor.advise(event)
    seconds = time.perf_counter() - started
    model = advisor._model
    tokens = len(model.tokenize(text.encode())) if model is not None and advisor.last_source == "local_model" else None
    metadata = getattr(model, "metadata", {}) if model is not None else {}
    peak_mib, memory_reason = peak_memory()
    print(json.dumps({"host": platform.platform(), "machine": platform.machine(),
                      "source": advisor.last_source, "model_path": advisor.model_path,
                      "model_name": metadata.get("general.name"),
                      "quantization": metadata.get("general.file_type"),
                      "load_seconds": load_seconds, "advice_seconds": seconds,
                      "peak_rss_mib": peak_mib, "peak_memory_unavailable_reason": memory_reason,
                      "output_tokens": tokens, "tokens_per_second": tokens / seconds if tokens else None},
                     indent=2))


if __name__ == "__main__":
    main()
