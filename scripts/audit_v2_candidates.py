#!/usr/bin/env python3
"""Offline validation of the unfrozen v2 task candidates; never calls an API."""

from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tasks.generator import task_set_checksum
from src.tasks.v2 import audit_v2_candidates, generate_v2_candidates, split_v2_candidates


def main() -> None:
    tasks = generate_v2_candidates()
    development, held_out = split_v2_candidates(tasks)
    report = {
        "status": "candidate_only_not_frozen_no_api_outcomes_observed",
        **audit_v2_candidates(tasks),
        "families": dict(sorted(Counter(task.task_family for task in tasks).items())),
        "checksum": task_set_checksum(tasks),
        "development": {
            **audit_v2_candidates(development),
            "families": dict(sorted(Counter(task.task_family for task in development).items())),
            "checksum": task_set_checksum(development),
        },
        "held_out_candidate": {
            **audit_v2_candidates(held_out),
            "families": dict(sorted(Counter(task.task_family for task in held_out).items())),
            "checksum": task_set_checksum(held_out),
        },
        "scope": {
            "confirmatory_channel": "T-H",
            "i_h_in_v2": False,
            "tcpc_role": "schema_oracle_upper_bound",
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
