"""Offline template renderer. External disclosure has not been approved."""

import logging

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
    return "\n".join(lines)


def render_llm(facts, client=None) -> str:
    """Return the template while external disclosure awaits owner approval.

    No client is invoked, even when provided. Enabling an external renderer
    requires a separate privacy decision and numeric-claim validation at
    the template's six-decimal precision before returning generated text.
    """
    logger.warning("External LLM rendering is disabled pending owner privacy approval")
    return render_template(facts)
