"""Facts-only rendering; external clients require explicit opt-in."""

import json
import logging
import os

from explain.numbers import numbers_supported

logger = logging.getLogger(__name__)

_REASONS = {
    "fully_served": "fully served",
    "critical_first": "critical-tier precedence",
    "deferred_lower_urgency": "deferred behind earlier urgency/priority groups",
    "shared_proportionally": "proportional sharing within the eligible group",
    "no_demand": "no demand",
    "missed_window": "missed window reported",
}


def render_template(facts) -> str:
    """Describe every agent using six decimal places for electrical kW."""
    lines = [f"{facts['timestamp']}: available {facts['available_kw']:.6f} kW; "
             f"demand {facts['total_demand_kw']:.6f} kW; "
             f"curtailed {facts['curtailed_kw']:.6f} kW."]
    for agent in facts["agents"]:
        lines.append(f"{agent['agent_id']}: {agent['allocated_kw']:.6f} of "
                     f"{agent['demand_kw']:.6f} kW served; {_REASONS[agent['reason']]}.")
    for factor in facts.get("surrogate_factors", []):
        lines.append(f"Surrogate factor: {factor['feature']} = {factor['value']:.6f}; "
                     f"contribution {factor['shap_value']:.6f} to predicted served fraction.")
    return "\n".join(lines)


def render_llm(facts, client=None, *, enabled=False, model=None) -> str:
    """Use an injected ``client(prompt, model=...)`` only when enabled.

    No SDK, credentials or remote endpoint is selected implicitly. The caller
    must explicitly approve disclosure with enabled=True and supply a client.
    Model configuration comes from the argument or AGENTGRID_EXPLAIN_MODEL.
    Exceptions and unsupported numeric claims fall back to the offline text.
    """
    if not enabled:
        logger.warning("External LLM rendering is disabled pending owner privacy approval")
        return render_template(facts)
    try:
        if client is None:
            raise ValueError("an explicit client is required")
        prompt = ("Explain only these computed allocation facts in two to four sentences. "
                  "Do not infer readings or causes. Use no numbers absent from the JSON.\n"
                  + json.dumps(facts, sort_keys=True, default=str))
        text = client(prompt, model=model or os.environ.get("AGENTGRID_EXPLAIN_MODEL"))
        if not numbers_supported(text, facts):
            raise ValueError("unsupported numeric claim or empty response")
        return text
    except Exception as exc:
        logger.warning("LLM explanation rejected (%s); using template", type(exc).__name__)
        return render_template(facts)
