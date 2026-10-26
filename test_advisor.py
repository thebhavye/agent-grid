"""Offline fault advice and numeric validation, with no GGUF downloads."""

from copy import deepcopy
from unittest.mock import Mock

import pytest

from dashboard.data import RunKey, run
from digital_twin.faults import DEVICE_FAULT_TYPES, FAULT_TYPES
from edge_llm.advisor import ADVICE, FaultAdvisor, summarize_faults


def event(kind="spike"):
    return {"fault_type": kind, "device": "battery_soc_pct", "readings": [],
            "start_step": 10, "end_step": 12, "coincident_unmet_kwh": .5}


@pytest.mark.parametrize("kind", sorted(DEVICE_FAULT_TYPES | FAULT_TYPES))
def test_fallback_covers_every_fault(kind, tmp_path):
    advisor = FaultAdvisor(tmp_path / "absent.gguf")
    assert advisor.advise(event(kind)) == ADVICE[kind]
    assert advisor.last_source == "fallback"


def test_local_model_is_optional_and_number_checked():
    model = Mock(return_value={"choices": [{"text": "Check the spike. Compare the observed 0.5 reading."}]})
    advisor = FaultAdvisor(model=model)
    assert "0.5" in advisor.advise(event())
    assert advisor.last_source == "local_model"
    model.return_value = {"choices": [{"text": "The battery reached 999 degrees."}]}
    assert advisor.advise(event()) == ADVICE["spike"]
    assert advisor.last_source == "fallback"
    model.side_effect = RuntimeError("offline")
    assert advisor.advise(event()) == ADVICE["spike"]


def test_windows_are_contiguous_and_unmet_is_not_claimed_causal():
    records = run(RunKey("clear_day", "hybrid", "inverter_trip"))
    before = deepcopy(records)
    events = summarize_faults(records)
    assert records == before
    assert len(events) == 1
    fault = events[0]
    assert (fault["start_step"], fault["end_step"]) == (144, 156)
    assert fault["end_min"] - fault["start_min"] == 60
    expected = sum(max(0, agent["demand_kw"] - agent["allocated_kw"]) * r["dt_min"] / 60
                   for r in records[144:156] for name, agent in r["agents"].items()
                   if name not in ("battery_1", "solar_1"))
    assert fault["coincident_unmet_kwh"] == pytest.approx(expected)
    records[150]["device_faults"] = []
    assert len(summarize_faults(records)) == 2


def test_sensor_used_values_and_nan_are_json_safe():
    import json
    records = run(RunKey("clear_day", "hybrid", "soc_sensor_dropout"))
    events = summarize_faults(records)
    assert events[0]["readings"][0]["used"]
    assert events[0]["readings"][0]["observed"] is None
    json.dumps(events, allow_nan=False)


def test_loading_failure_is_cached(tmp_path, monkeypatch):
    import sys
    model_path = tmp_path / "fake.gguf"
    model_path.write_text("not a model")
    factory = Mock(side_effect=RuntimeError("bad model"))
    monkeypatch.setitem(sys.modules, "llama_cpp", Mock(Llama=factory))
    advisor = FaultAdvisor(model_path)
    factory.assert_not_called()
    advisor.advise(event())
    advisor.advise(event())
    factory.assert_called_once()


@pytest.mark.parametrize("accepted", [False, True])
def test_benchmark_only_counts_accepted_model_output(monkeypatch, capsys, accepted):
    import json
    import sys
    from edge_llm import benchmark
    model = Mock(metadata={})
    model.tokenize.return_value = [1, 2, 3]
    advisor = Mock(_model=model, model_path="fake.gguf",
                   last_source="local_model" if accepted else "fallback")
    advisor.advise.return_value = "Check inverter alarms."
    monkeypatch.setattr(benchmark, "FaultAdvisor", lambda path: advisor)
    monkeypatch.setattr(sys, "argv", ["benchmark"])
    benchmark.main()
    report = json.loads(capsys.readouterr().out)
    assert report["output_tokens"] == (3 if accepted else None)
    if accepted:
        assert report["tokens_per_second"] > 0
    else:
        assert report["tokens_per_second"] is None
        model.tokenize.assert_not_called()


def test_benchmark_without_resource_uses_windows_peak(monkeypatch, capsys):
    import json
    import sys
    from edge_llm import benchmark
    monkeypatch.setitem(sys.modules, "resource", None)
    import importlib
    importlib.reload(benchmark)
    monkeypatch.setitem(sys.modules, "psutil", Mock(Process=lambda: Mock(
        memory_info=lambda: Mock(peak_wset=128 * 1024 ** 2))))
    monkeypatch.setattr(sys, "argv", ["benchmark"])
    benchmark.main()
    report = json.loads(capsys.readouterr().out)
    assert report["peak_rss_mib"] == 128
    assert report["peak_memory_unavailable_reason"] is None


def test_benchmark_memory_unavailable_does_not_crash(monkeypatch, capsys):
    import json
    import sys
    from edge_llm import benchmark
    monkeypatch.setitem(sys.modules, "resource", None)
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(sys, "argv", ["benchmark"])
    benchmark.main()
    report = json.loads(capsys.readouterr().out)
    assert report["peak_rss_mib"] is None
    assert "unavailable" in report["peak_memory_unavailable_reason"]
