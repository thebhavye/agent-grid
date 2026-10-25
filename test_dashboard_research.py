"""Optional Streamlit interactions for research panels (no model downloads)."""

import pytest


def test_dashboard_research_and_fault_advice():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    pytest.importorskip("shap")
    pytest.importorskip("torch")
    from streamlit.testing.v1 import AppTest
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        app = AppTest.from_file("dashboard/app.py", default_timeout=60).run()
        assert not app.exception
        next(c for c in app.checkbox if c.label == "Battery health experiment").check().run()
        assert not app.exception
        assert "Battery health" in [h.value for h in app.subheader]
        next(c for c in app.checkbox if c.label == "Show optional SHAP surrogate").check().run()
        assert not app.exception
        assert "Why (SHAP)" in [h.value for h in app.subheader]
        next(s for s in app.selectbox if s.label == "Inject a fault").select("inverter_trip").run()
        assert not app.exception
        assert any("Source: fallback" in c.value for c in app.caption)


def test_advisor_resource_and_advice_survive_widget_reruns(monkeypatch):
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    from unittest.mock import Mock
    import edge_llm

    st.cache_data.clear()
    st.cache_resource.clear()
    advisor = Mock(last_source="local_model")
    advisor.advise.return_value = "Inspect the inverter. Check its alarms."
    factory = Mock(return_value=advisor)
    monkeypatch.setattr(edge_llm, "FaultAdvisor", factory)
    app = AppTest.from_file("dashboard/app.py", default_timeout=60).run()
    next(s for s in app.selectbox if s.label == "Inject a fault").select("inverter_trip").run()
    assert not app.exception
    assert factory.call_count == advisor.advise.call_count == 1
    app.select_slider[0].set_value(100).run()
    assert not app.exception
    assert factory.call_count == advisor.advise.call_count == 1
    assert any("Source: local_model" in c.value for c in app.caption)
    next(s for s in app.selectbox if s.label == "Inject a fault").select("charger_fault").run()
    assert not app.exception
    assert factory.call_count == 1
    assert advisor.advise.call_count == 2
    st.cache_data.clear()
    st.cache_resource.clear()
