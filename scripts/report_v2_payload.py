#!/usr/bin/env python3
"""Report the oracle-removable notes share without any API call."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.accounting.tokenizer import resolve_tiktoken
from src.analysis.v2 import non_contract_payload_fraction
from src.environment.tools import DeterministicToolEnvironment
from src.experiments.config import load_mapping_yaml
from src.tasks.v2 import generate_v2_candidates, split_v2_candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model = load_mapping_yaml("configs/model.yaml")
    tokenizer = resolve_tiktoken(model["model_id"], proxy_encoding=model["tokenizer"].get("proxy_encoding"))
    tasks, held_out = split_v2_candidates(generate_v2_candidates())
    report = {}
    for split, selected in (("development", tasks), ("held_out_candidate", held_out)):
        values = defaultdict(list)
        for task in selected:
            environment = DeterministicToolEnvironment(task.hidden_state)
            first = task.expected_tool_calls[0]
            output = environment.execute(first["name"], first["arguments"]).result
            values[task.task_family].append(non_contract_payload_fraction(output, tokenizer))
        report[split] = {
            family: {
                "pairs": len(fractions),
                "mean_notes_token_share": statistics.fmean(fractions),
                "min_notes_token_share": min(fractions),
                "max_notes_token_share": max(fractions),
            }
            for family, fractions in sorted(values.items())
        }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
