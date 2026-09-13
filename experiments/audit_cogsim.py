"""Offline, adversarial checks for using the frozen Cog-Sim in branching RL.

Run from the CSTPO workspace: python experiments/audit_cogsim.py
Failures are recorded observations, not fixes to the frozen simulator.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Cog-Sim"
sys.path.insert(0, str(SOURCE))

from evaluation.robustness import run as robustness
from evaluation.state_transition import rj_violations, state_stability
from simulator.run_sim import SCENARIOS, build_state
from simulator.simulator import UserSimulator
from simulator.state.schema import BDIItem, MAX_ITEMS, UserState
from simulator.state.updater import constrained_apply


def fingerprint():
    return {str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((SOURCE / "simulator").rglob("*.py"))}


def run_checks():
    checks = []

    def record(name, expected, observed, ok):
        checks.append(dict(name=name, expected=expected, observed=observed, ok=bool(ok)))

    scenario = copy.deepcopy(SCENARIOS["donation"])
    left, right = build_state(scenario), build_state(scenario)
    initial = right.beliefs[0].strength
    left.beliefs[0].strength = 0.123
    record("build_state_isolates_repeated_initialization", initial,
           right.beliefs[0].strength, right.beliefs[0].strength == initial)

    left = build_state(copy.deepcopy(SCENARIOS["donation"]))
    right = copy.deepcopy(left)
    left.beliefs[0].strength = 0.123
    record("deepcopy_branch_isolation", 3.0, right.beliefs[0].strength,
           right.beliefs[0].strength == 3.0)

    for duplicate in (False, True):
        state = UserState(beliefs=[BDIItem("B1", "belief", "anchor", 1.0)])
        updates = [{"id": "B1", "new_strength": 2.0}]
        if duplicate:
            updates.append({"id": "B1", "new_strength": 3.0})
        constrained_apply(state, {"bdi_updates": updates}, "central", "accept", [])
        delta = state.find("B1").strength - 1.0
        record("duplicate_updates_turn_cap" if duplicate else "single_update_turn_cap",
               "abs(delta) <= 1.0", delta, abs(delta) <= 1.0 + 1e-9)

    state = UserState(beliefs=[BDIItem("B1", "belief", "anchor", 0.3,
                                     core=False, active=False)])
    constrained_apply(state, {"bdi_updates": [{"id": "B1", "new_strength": 0.8}]},
                      "peripheral", "noncommit", [])
    item = state.find("B1")
    record("reactivation_checked_after_clipping", "inactive or strength >= 0.7",
           asdict(item), not item.active or item.strength >= 0.7 - 1e-9)

    state = UserState(beliefs=[BDIItem(f"B{i}", "belief", str(i), strength, core=False)
                              for i, strength in enumerate([0.8, 1.5, 2.0, 3.0], 1)])
    constrained_apply(state, {"new_items": [{"type": "belief", "content": "incoming",
                                            "strength": 0.9, "core": False}]},
                      "central", "accept", [])
    record("evict_weakest_noncore", "replace 0.8 with 0.9",
           [(b.content, b.strength) for b in state.beliefs],
           state.find("B1") is None and any(b.content == "incoming" for b in state.beliefs))

    state = UserState(beliefs=[BDIItem(f"B{i}", "belief", str(i), 1.0) for i in range(1, 5)]
                     + [BDIItem("B5", "belief", "inactive", 0.4, core=False, active=False)])
    constrained_apply(state, {"bdi_updates": [{"id": "B5", "new_strength": 0.8}]},
                      "central", "accept", [])
    count = sum(b.active for b in state.beliefs)
    record("reactivation_respects_active_capacity", "active beliefs <= 4", count,
           count <= MAX_ITEMS["belief"])

    # Inject at the engine boundary, so the real simulate_turn guards also execute.
    from unittest.mock import patch
    import simulator.simulator as orchestrator
    state = build_state(copy.deepcopy(SCENARIOS["donation"]))
    sim = UserSimulator(state, llm=object())
    with patch.object(orchestrator, "classify_mode", return_value=("influence", None)), \
         patch.object(orchestrator, "extract_route_features", return_value={
             "relevance": 0.8, "argument_strength": 0.8, "cue_strength": 0.2,
             "target_proposition": "anchor"}), \
         patch.object(orchestrator, "estimate_discrepancy", return_value={
             "target": "anchor", "stance_distance": 0.2, "relevant_state_ids": ["B1"]}), \
         patch.object(orchestrator, "propose_cognitive_update", return_value={
             "bdi_updates": [1], "new_items": []}):
        try:
            sim.simulate_turn("probe")
            error = None
        except Exception as exc:
            error = type(exc).__name__
    record("nested_invalid_update_does_not_crash_turn", None, error, error is None)

    sim = UserSimulator(build_state(copy.deepcopy(SCENARIOS["donation"])), llm=object())
    sim.prev_gc = 0.8
    restored = UserSimulator(copy.deepcopy(sim.state), llm=object())
    record("state_only_checkpoint_preserves_gc_memory", sim.prev_gc, restored.prev_gc,
           sim.prev_gc == restored.prev_gc)
    # The preceding failure is an integration requirement, not a promised snapshot API.
    checks[-1]["category"] = "integration_requirement"

    before = build_state(copy.deepcopy(SCENARIOS["donation"])).bdi_dict()
    after = copy.deepcopy(before)
    after["beliefs"][0]["content"] = "a different proposition"
    reported = state_stability(SimpleNamespace(mode="elicit", bdi_before=before, bdi_after=after))
    record("stability_audit_detects_content_change", "nonempty violations", reported, bool(reported))
    checks[-1]["category"] = "audit_coverage"
    return checks


def historical_logs():
    rows = []
    for path in sorted((SOURCE / "runs").glob("*_log.jsonl")):
        logs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        violations = []
        duplicate_proposals = 0
        for row in logs:
            log = SimpleNamespace(**copy.deepcopy(row))
            violations.extend(rj_violations(log))
            updates = (row.get("proposed_bdi") or {}).get("bdi_updates", [])
            ids = [u.get("id") for u in updates if isinstance(u, dict)]
            duplicate_proposals += len(ids) != len(set(ids))
        rows.append(dict(file=str(path.relative_to(ROOT)), turns=len(logs),
                         rj_violations=violations, duplicate_proposal_turns=duplicate_proposals))
    return rows


def main():
    base = robustness()
    checks = run_checks()
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  scope="Offline edge cases and existing demo logs; no model calls or human validation",
                  source_sha256=fingerprint(), existing_robustness=base,
                  branching_checks=checks, historical_demo_logs=historical_logs())
    destination = ROOT / "experiments/results/offline_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Existing robustness: {base['passed']}/{base['total']}")
    for check in checks:
        print(f"{'PASS' if check['ok'] else 'FAIL'} {check['name']}: {check['observed']}")
    print(f"Saved {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
