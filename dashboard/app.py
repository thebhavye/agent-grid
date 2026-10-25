"""AgentGrid dashboard: run with `streamlit run dashboard/app.py`."""

import sys
import os
from threading import Lock
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

# Allow `streamlit run dashboard/app.py` from the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import data  # noqa: E402
from scenarios import CATALOG, list_scenarios  # noqa: E402

# Validated categorical slots (dataviz reference palette, fixed order) and chrome.
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
CRITICAL, WARNING = "#d03b3b", "#fab219"
MUTED, GRID, INK = "#898781", "#e1e0d9", "#52514e"
GROUP_COLORS = dict(zip(data.GROUPS, (BLUE, ORANGE, AQUA, VIOLET)))

st.set_page_config(page_title="AgentGrid Control Room", page_icon="⚡", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
      [data-testid="stMetricValue"] {font-size: 1.55rem;}
      .explain {border-left: 3px solid #2a78d6; padding: .2rem 0 .2rem .8rem; margin: .3rem 0;}
      .explain p {margin: .15rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_run(scenario, strategy, fault, policy, battery_health=False):
    return data.run(data.RunKey(scenario, strategy, fault, policy, battery_health))


@st.cache_resource(show_spinner="Fitting offline surrogate…")
def load_surrogate():
    from explain.shap_surrogate import evaluate, scenario_frame, train_surrogate
    model = train_surrogate(scenario_frame())
    return model, evaluate(model, scenario_frame(("heatwave",), (13,)))


@st.cache_resource(show_spinner="Training battery-health model on saved scenarios…")
def load_health_model():
    from battery_health.dataset import generate_battery_dataset
    from battery_health.pinn import train
    return train(generate_battery_dataset(("clear_day", "monsoon_cloudy"), days=3))



@st.cache_data(show_spinner=False)
def load_comparison(scenario, fault, policy):
    return data.comparison(scenario, fault, policy)


@st.cache_resource(show_spinner=False)
def load_advisor(model_path):
    from edge_llm import FaultAdvisor
    return FaultAdvisor(model_path=model_path), Lock()


@st.cache_data(show_spinner=False)
def event_advice(model_path, event):
    advisor, lock = load_advisor(model_path)
    with lock:
        advice = advisor.advise(event)
        return advice, advisor.last_source


def base_layout(fig, title, y_title, height=320):
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=15)),
        height=height, margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified", legend=dict(orientation="h", y=-0.18, x=0, xanchor="left", yanchor="top"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=MUTED),
                     tickformat="%H:%M<br>%d %b", hoverformat="%d %b %H:%M")
    fig.update_yaxes(title=y_title, gridcolor=GRID, zeroline=True, zerolinecolor=GRID,
                     tickfont=dict(color=MUTED), rangemode="tozero")
    return fig


def mark_step(fig, frame, step):
    fig.add_vline(x=frame.time.iloc[step], line_width=1, line_dash="dot", line_color=MUTED)
    return fig


def power_chart(frame, step):
    fig = go.Figure()
    x = frame.time
    fig.add_trace(go.Scatter(x=x, y=frame.solar_kw, name="Solar generation", fill="tozeroy",
                             fillcolor="rgba(237,161,0,0.18)", line=dict(color=YELLOW, width=2)))
    fig.add_trace(go.Scatter(x=x, y=frame.delivered_kw, name="Delivered to loads",
                             line=dict(color=BLUE, width=2)))
    fig.add_trace(go.Scatter(x=x, y=frame.battery_discharge_kw, name="Battery discharge",
                             line=dict(color=AQUA, width=2)))
    fig.add_trace(go.Scatter(x=x, y=frame.battery_charge_kw, name="Battery charge",
                             line=dict(color=AQUA, width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=x, y=frame.curtailed_kw, name="Curtailed solar",
                             line=dict(color=MUTED, width=1.5, dash="dash")))
    fig.update_traces(hovertemplate="%{y:.2f} kW")
    return mark_step(base_layout(fig, "Power flows", "kW", height=360), frame, step)


def soc_chart(frame, step):
    fig = go.Figure()
    x = frame.time
    plugged = frame.ev_soc_pct.where(frame.ev_plugged_in)
    fig.add_trace(go.Scatter(x=x, y=frame.battery_soc_pct, name="Home battery SOC",
                             line=dict(color=AQUA, width=2), hovertemplate="%{y:.1f}%"))
    fig.add_trace(go.Scatter(x=x, y=frame.ev_soc_pct, name="EV SOC (away)",
                             line=dict(color=ORANGE, width=1.5, dash="dot"), hovertemplate="%{y:.1f}%"))
    fig.add_trace(go.Scatter(x=x, y=plugged, name="EV SOC (plugged in)",
                             line=dict(color=ORANGE, width=2.5), hovertemplate="%{y:.1f}%"))
    fig.add_trace(go.Scatter(x=x, y=frame.battery_reserve_pct, name="Battery reserve",
                             line=dict(color=MUTED, width=1, dash="dash"), hoverinfo="skip"))
    fig = base_layout(fig, "State of charge", "%", height=360)
    fig.update_yaxes(range=[0, 102])
    return mark_step(fig, frame, step)


def allocation_chart(groups, frame, step):
    fig = go.Figure()
    for group in data.GROUPS:
        part = groups[groups.group == group]
        fig.add_trace(go.Scatter(x=part.time, y=part.kw, name=group, stackgroup="alloc",
                                 line=dict(width=0.5, color="#fcfcfb"), fillcolor=GROUP_COLORS[group],
                                 hovertemplate="%{y:.2f} kW"))
    return mark_step(base_layout(fig, "Where the power went", "kW", height=360), frame, step)


def unmet_chart(frame, step):
    fig = go.Figure()
    x = frame.time
    fig.add_trace(go.Scatter(x=x, y=frame.critical_unmet_kw, name="Critical unmet (priority 1-2)",
                             stackgroup="unmet", line=dict(width=0.5, color="#fcfcfb"),
                             fillcolor=CRITICAL, hovertemplate="%{y:.2f} kW"))
    fig.add_trace(go.Scatter(x=x, y=frame.flexible_unmet_kw, name="Flexible unmet (priority 3-5)",
                             stackgroup="unmet", line=dict(width=0.5, color="#fcfcfb"),
                             fillcolor=WARNING, hovertemplate="%{y:.2f} kW"))
    return mark_step(base_layout(fig, "Unmet demand", "kW", height=360), frame, step)


KPI_TILES = {
    "critical_unmet_kwh": "Critical unmet · kWh",
    "flexible_unmet_kwh": "Flexible unmet · kWh",
    "ev_shortfall_pct_at_departure": "EV shortfall · pp",
    "curtailed_solar_kwh": "Curtailed solar · kWh",
    "battery_discharge_kwh": "Battery used · kWh",
    "battery_reserve_violation_steps": "Reserve breaches",
}


def kpi_row(current, other, strategy_label, other_label):
    cols = st.columns(3) + st.columns(3)  # two rows of three stay readable on a laptop
    for col, (key, title) in zip(cols, KPI_TILES.items()):
        delta = current[key] - other[key]
        same = abs(delta) < 5e-3
        col.metric(
            title, f"{current[key]:.2f}",
            delta=None if same else f"{delta:+.2f} vs {other_label.split()[0].lower()}",
            delta_color="inverse" if key in data.LOWER_IS_BETTER else "off",
            help=f"{data.METRIC_LABELS[key]} for the {strategy_label.lower()}; the delta "
                 f"compares with the {other_label.lower()} on the same scenario.",
        )


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚡ AgentGrid")
    st.caption("Multi-agent power allocation for a residential microgrid")
    scenario = st.selectbox("Scenario", list_scenarios(),
                            index=list_scenarios().index("demo_sunny_home"),
                            format_func=lambda s: s.replace("_", " ").capitalize())
    st.caption(data.SCENARIO_NOTES.get(scenario, ""))
    strategy = st.radio("Allocation strategy", list(data.STRATEGIES),
                        format_func=lambda s: data.STRATEGIES[s][0])
    fault = st.selectbox("Inject a fault", list(data.FAULT_PRESETS),
                         format_func=lambda f: data.FAULT_PRESETS[f][0])
    policy = "hold_last"
    if data.FAULT_PRESETS[fault][1] is not None:
        policy = st.radio("Bad sensor reading policy", ["hold_last", "exclude"],
                          format_func={"hold_last": "Hold last good value",
                                       "exclude": "Exclude the agent"}.get)
    health_enabled = st.checkbox("Battery health experiment", value=False)
    st.divider()
    st.caption("Closed-loop digital twin · offline saved Vellore weather · "
               "every step checked for power balance")

other = "baseline" if strategy == "hybrid" else "hybrid"
with st.spinner("Running the closed-loop simulation…"):
    records = load_run(scenario, strategy, fault, policy, health_enabled)
    other_records = load_run(scenario, other, fault, policy, health_enabled)
start_date = CATALOG[scenario]["start_date"]
frame = data.timeseries(records, start_date)
groups = data.allocation_groups(records, start_date)
current, other_metrics = data.metrics(records), data.metrics(other_records)
label, other_label = data.STRATEGIES[strategy][0], data.STRATEGIES[other][0]

# ---------------------------------------------------------------- header + KPIs
st.title("AgentGrid Control Room")
st.markdown(f"**{scenario.replace('_', ' ').capitalize()}** · {label} · "
            f"{data.FAULT_PRESETS[fault][0]} · {len(records)} five-minute steps")
kpi_row(current, other_metrics, label, other_label)

# ---------------------------------------------------------------- step selector
step = st.select_slider("Inspect a moment", options=list(range(len(frame))),
                        value=min(222, len(frame) - 1),
                        format_func=lambda i: frame.clock.iloc[i])
record = records[step]

overview, agents_tab, compare_tab, faults_tab = st.tabs(
    ["Overview", "Agents & decisions", "Hybrid vs baseline", "Faults"])

with overview:
    left, right = st.columns(2)
    left.plotly_chart(power_chart(frame, step), width="stretch")
    right.plotly_chart(soc_chart(frame, step), width="stretch")
    left, right = st.columns(2)
    left.plotly_chart(allocation_chart(groups, frame, step), width="stretch")
    right.plotly_chart(unmet_chart(frame, step), width="stretch")

with agents_tab:
    row = frame.iloc[step]
    a, b, c, d = st.columns(4)
    a.metric("Solar", f"{row.solar_kw:.2f} kW")
    battery_text = {"charge": f"Charging {record['battery_kw']:.2f} kW",
                    "discharge": f"Supplying {-record['battery_kw']:.2f} kW",
                    "idle": "Idle"}[row["mode"]]
    b.metric("Battery", battery_text)
    c.metric("Battery SOC", f"{row.battery_soc_pct:.1f}%")
    d.metric("EV plugged in" if row.ev_plugged_in else "EV away", f"{row.ev_soc_pct:.1f}%")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Bids at this moment")
        table = data.step_table(record)
        st.dataframe(
            table, hide_index=True, width="stretch",
            column_config={
                "Urgency": st.column_config.ProgressColumn("Urgency", min_value=0, max_value=1, format="%.2f"),
                "Served (%)": st.column_config.ProgressColumn("Served (%)", min_value=0, max_value=100, format="%.0f%%"),
                "Priority": st.column_config.NumberColumn("Priority", help="1 = critical … 5 = deferrable"),
            },
        )
    with right:
        st.subheader("Why this allocation")
        lines = data.explain_step(record, strategy)
        st.markdown('<div class="explain">' + "".join(f"<p>{line}</p>" for line in lines) + "</div>",
                    unsafe_allow_html=True)
        st.caption("Generated from the arbiter's own rules (facts-first explainer); "
                   "no numbers are invented.")

with compare_tab:
    st.subheader("Same scenario, same weather, two strategies")
    st.caption("Each run uses an identical twin; only the allocation rule changes. "
               "Lower is better for unmet energy, shortfall, curtailment and violations.")
    table = load_comparison(scenario, fault, policy)
    show = table.drop(columns="key")
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(c, format="%.3f")
                                for c in ("Hybrid", "Baseline", "Hybrid - baseline")})
    picked = table[table.key.isin(["critical_unmet_kwh", "flexible_unmet_kwh",
                                   "curtailed_solar_kwh", "battery_discharge_kwh"])]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=picked.Metric, x=picked.Hybrid, name="Hybrid arbiter",
                         orientation="h", marker_color=BLUE, hovertemplate="%{x:.2f}"))
    fig.add_trace(go.Bar(y=picked.Metric, x=picked.Baseline, name="Proportional baseline",
                         orientation="h", marker_color=ORANGE, hovertemplate="%{x:.2f}"))
    fig = base_layout(fig, "Energy outcomes (kWh)", "", height=300)
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08, hovermode="y unified")
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title=None, autorange="reversed")
    st.plotly_chart(fig, width="stretch")

with faults_tab:
    if fault == "none":
        st.info("No fault injected. Pick one in the sidebar to see how the runner copes.")
    else:
        st.subheader(data.FAULT_PRESETS[fault][0])
        events = frame[frame.faults != ""][["clock", "faults", "solar_kw", "battery_soc_pct", "mode"]]
        st.caption(f"{len(events)} affected steps. Power balance still holds on every step "
                   f"(max error {current['energy_balance_max_error_kw']:.1e} kW).")
        st.dataframe(events.rename(columns={"clock": "Time", "faults": "Fault", "solar_kw": "Solar (kW)",
                                            "battery_soc_pct": "Battery SOC (%)", "mode": "Battery mode"}),
                     hide_index=True, width="stretch")
        subs = [dict(time=frame.clock.iloc[i], **s) for i, r in enumerate(records)
                for s in r.get("substitutions", [])]
        if subs:
            st.markdown("**Sensor substitutions made by the runner**")
            st.dataframe(subs, hide_index=True, width="stretch")


with agents_tab:
    if strategy == "hybrid" and fault == "none" and st.checkbox("Show optional SHAP surrogate"):
        try:
            from explain.shap_surrogate import explain_step as shap_step
            model, fidelity = load_surrogate()
            candidates = [name for name, agent in record["agents"].items()
                          if agent["demand_kw"] > 0 and name != "solar_1"
                          and (name != "battery_1" or record["mode"] == "charge")]
            if candidates:
                selected_agent = st.selectbox("Agent for SHAP", candidates, format_func=data.agent_label)
                result = shap_step(model, record, selected_agent)
                st.subheader("Why (SHAP)")
                factors = result["top_factors"]
                fig = go.Figure(go.Bar(x=[f["shap_value"] for f in factors],
                                       y=[f["feature"] for f in factors], orientation="h"))
                st.plotly_chart(fig, width="stretch")
                st.caption(f"Hybrid surrogate, heatwave holdout R² {fidelity['r2']:.3f}, "
                           f"MAE {fidelity['mae']:.3f}. Contributions explain the model prediction, "
                           "not causal effects.")
                from explain.render import render_template
                st.text(render_template({"timestamp": record["timestamp"],
                                         "available_kw": record["available_kw"],
                                         "total_demand_kw": sum(record["demand_kw"].values()),
                                         "curtailed_kw": record["curtailed_kw"],
                                         "agents": [], "surrogate_factors": factors}))
            else:
                st.info("No positive-demand agent at this moment.")
        except ImportError:
            st.info("Install requirements-research.txt to enable SHAP explanations.")

with faults_tab:
    st.subheader("Advisor")
    from edge_llm import summarize_faults
    model_path = os.environ.get("AGENTGRID_GGUF_MODEL")
    advice_events = summarize_faults(records)
    if not advice_events:
        st.caption("No labelled fault events in this run.")
    for event in advice_events:
        advice, source = event_advice(model_path, event)
        st.markdown(f"**{event['fault_type']} · {event['device']} · "
                    f"steps {event['start_step']}–{event['end_step']} (end exclusive)**")
        st.write(advice)
        st.caption(f"Source: {source}. Coincident unmet energy "
                   f"{event['coincident_unmet_kwh']:.3f} kWh; this is not causal fault impact.")

with overview:
    if health_enabled:
        st.subheader("Battery health")
        try:
            import pandas as pd
            from battery_health.dataset import operation_row
            from battery_health.pinn import thermal_risk
            from digital_twin.battery_model import BatteryModelConfig
            config = BatteryModelConfig(degradation=True, capacity_kwh=CATALOG[scenario]["capacity_kwh"])
            throughput = 0.0
            rows = []
            for item in records:
                rows.append(operation_row(item, config, throughput))
                throughput += abs(item["battery_kw"]) * item["dt_min"] / 60
            battery_frame = pd.DataFrame(rows)
            prediction = load_health_model().predict(battery_frame)
            chart = pd.DataFrame({"Time": frame.time, "Predicted SoH (%)": prediction[:, 0],
                                  "Twin SoH (%)": battery_frame.next_health_pct}).set_index("Time")
            st.line_chart(chart)
            temp_chart = pd.DataFrame({"Time": frame.time, "Predicted temperature (°C)": prediction[:, 1],
                                      "Twin temperature (°C)": battery_frame.next_temperature_c}).set_index("Time")
            st.line_chart(temp_chart)
            selected = battery_frame.iloc[step]
            flags = thermal_risk(prediction[step, 1], selected.temperature_c, selected.dt_min)
            st.write("Risk flags at selected moment: " + (", ".join(flags) if flags else "none"))
            st.caption("Experimental one-step predictions conditioned on observed state; "
                       "assumed aging coefficients and demonstration thresholds, not validated battery diagnostics.")
        except ImportError:
            st.info("Install requirements-battery.txt for battery-health predictions.")
