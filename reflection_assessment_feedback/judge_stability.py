from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd

from reflection_assessment_feedback.judge_study import study_results


FEEDBACK_MEASUREMENT_COLUMNS = [
    "reflection_id", "packet_id", "condition", "generation_repetition", "judge_repetition",
    "central_actionable_n", "central_missing_n", "central_partial_n",
    "central_missing_rate", "central_full_coverage_rate", "any_major_issue",
    "any_major_contradiction", "overall_severity",
    "positive_eligible_n", "positive_addressed_n", "positive_rate_denominator_n",
    "positive_recognition_rate", "central_addressed_n", "central_rate_denominator_n",
]

MICRO_METRIC_COLUMNS = [
    "Generator", "generation_repetition", "judge_repetition", "PRR", "CMR", "CPCR", "CFCR", "MDI",
    "positive_addressed_n", "positive_denominator_n",
    "central_missing_n", "central_partial_n", "central_addressed_n", "central_denominator_n",
    "major_discrepancy_feedbacks_n", "evaluated_feedbacks_n",
]


def _percentage(numerator: int, denominator: int) -> float | None:
    return round(100 * numerator / denominator, 2) if denominator else None


def extract_feedback_measurements(directory: str | Path) -> dict[str, pd.DataFrame]:
    """Extract packet/repetition measures and the full frozen-concern audit table."""
    tables = analyze_frozen_concern_study(directory)
    return {
        "measurements": tables["measurements"],
        "concern_coverage": tables["concern_coverage"],
    }


def summarize_feedback_metrics(measurements: pd.DataFrame) -> pd.DataFrame:
    """Macro-average reflection rates without pooling independent repetitions."""
    keys = ["condition", "generation_repetition", "judge_repetition"]
    if measurements.duplicated([*keys, "reflection_id"]).any():
        raise ValueError("Metric summaries require one packet per reflection within each repetition.")
    return measurements.groupby(keys).agg(
        reflections_n=("reflection_id", "nunique"),
        prr_reflections_n=("positive_recognition_rate", "count"),
        positive_recognition_rate=("positive_recognition_rate", "mean"),
        cmr_reflections_n=("central_missing_rate", "count"),
        central_missing_rate=("central_missing_rate", "mean"),
        cfcr_reflections_n=("central_full_coverage_rate", "count"),
        central_full_coverage_rate=("central_full_coverage_rate", "mean"),
        evaluated_feedbacks_n=("any_major_issue", "count"),
        major_discrepancy_feedbacks_n=("any_major_issue", "sum"),
        major_discrepancy_incidence=("any_major_issue", "mean"),
    ).reset_index()


def summarize_micro_feedback_metrics(measurements: pd.DataFrame) -> pd.DataFrame:
    """Micro-averaged percentages per generator; counts are pooled across reflections only."""
    keys = ["condition", "generation_repetition", "judge_repetition"]
    if measurements.duplicated([*keys, "reflection_id"]).any():
        raise ValueError("Metric summaries require one packet per reflection within each repetition.")
    rows = []
    for (condition, generation_repetition, judge_repetition), group in measurements.groupby(keys, sort=True):
        # Unexecuted slots have null outcome counts and are excluded from every denominator.
        judged = group[group["any_major_issue"].notna()]
        counts = {
            column: int(judged[column].fillna(0).sum())
            for column in (
                "positive_addressed_n", "positive_rate_denominator_n", "central_missing_n",
                "central_partial_n", "central_addressed_n", "central_rate_denominator_n",
            )
        }
        central = counts["central_rate_denominator_n"]
        if counts["central_missing_n"] + counts["central_partial_n"] + counts["central_addressed_n"] != central:
            raise ValueError("Central coverage outcomes do not partition the eligible denominator.")
        major = int(judged["any_major_issue"].astype(bool).sum())
        rows.append({
            "Generator": condition, "generation_repetition": generation_repetition,
            "judge_repetition": judge_repetition,
            "PRR": _percentage(counts["positive_addressed_n"], counts["positive_rate_denominator_n"]),
            "CMR": _percentage(counts["central_missing_n"], central),
            "CPCR": _percentage(counts["central_partial_n"], central),
            "CFCR": _percentage(counts["central_addressed_n"], central),
            "MDI": _percentage(major, len(judged)),
            "positive_addressed_n": counts["positive_addressed_n"],
            "positive_denominator_n": counts["positive_rate_denominator_n"],
            "central_missing_n": counts["central_missing_n"],
            "central_partial_n": counts["central_partial_n"],
            "central_addressed_n": counts["central_addressed_n"],
            "central_denominator_n": central,
            "major_discrepancy_feedbacks_n": major, "evaluated_feedbacks_n": len(judged),
        })
    return pd.DataFrame(rows, columns=MICRO_METRIC_COLUMNS).astype(
        {"PRR": "Float64", "CMR": "Float64", "CPCR": "Float64", "CFCR": "Float64", "MDI": "Float64"},
    )


def analyze_feedback_alignment_study(directory: str | Path) -> dict[str, pd.DataFrame]:
    study, results = study_results(directory)
    protocol = study["protocol"]
    if protocol.get("feedback_prompt_version") != "0.4.0":
        raise ValueError("Feedback alignment analysis requires feedback prompt 0.4.0.")
    rows, stability = [], []
    for entry in protocol["entries"]:
        base = {
            "document_id": entry["identity"]["document_id"],
            "condition": entry["identity"]["condition"],
            "generation_repetition": entry["generation_repetition"],
            "packet_id": entry["packet"]["packet_id"],
        }
        verdicts = []
        for repetition in range(1, protocol["judge_repetitions"] + 1):
            record = results.get((base["packet_id"], repetition))
            output = record["judgment"]["result"] if record else None
            verdict = output["verdict"] if output else "missing"
            verdicts.append(verdict)
            rows.append({
                **base, "judge_repetition": repetition, "verdict": verdict,
                "rationale": output["rationale"] if output else None,
                "comparisons": output["comparisons"] if output else [],
            })
        complete = all(v not in {"missing", "unable_to_judge"} for v in verdicts)
        stability.append({
            **base, "expected_repetitions": protocol["judge_repetitions"],
            "missing": verdicts.count("missing"), "unable_to_judge": verdicts.count("unable_to_judge"),
            "unanimous": len(set(verdicts)) == 1 if complete and len(verdicts) > 1 else None,
            "pairwise_agreements": sum(a == b for a, b in combinations(verdicts, 2)) if complete else None,
            "pairwise_comparisons": len(list(combinations(verdicts, 2))) if complete else None,
        })
    return {"decisions": pd.DataFrame(rows), "stability": pd.DataFrame(stability)}


def analyze_feedback_issue_study(directory: str | Path) -> dict[str, pd.DataFrame]:
    study, results = study_results(directory)
    if study["protocol"].get("feedback_prompt_version") not in {"0.5.0", "0.6.0", "0.7.0"}:
        raise ValueError("Feedback issue analysis requires feedback prompt 0.5.0, 0.6.0 or 0.7.0.")
    decisions, issues = [], []
    for entry in study["protocol"]["entries"]:
        for repetition in range(1, study["protocol"]["judge_repetitions"] + 1):
            base = {
                "document_id": entry["identity"]["document_id"],
                "condition": entry["identity"]["condition"],
                "generation_repetition": entry["generation_repetition"],
                "packet_id": entry["packet"]["packet_id"], "judge_repetition": repetition,
            }
            record = results.get((base["packet_id"], repetition))
            output = record["judgment"]["result"] if record else None
            decisions.append({
                **base, "overall_severity": output["overall_severity"] if output else "missing",
                "rationale": output["rationale"] if output else None,
                "reference_concerns": output["reference_concerns"] if output else [],
                "reference_uncertainties": output["reference_uncertainties"] if output else [],
                "major_issues": sum(i["severity"] == "major" for i in output["issues"]) if output else None,
                "minor_issues": sum(i["severity"] == "minor" for i in output["issues"]) if output else None,
            })
            if output:
                issues.extend({**base, **issue} for issue in output["issues"])
    return {"decisions": pd.DataFrame(decisions), "issues": pd.DataFrame(issues)}


def analyze_frozen_concern_study(directory: str | Path) -> dict[str, pd.DataFrame]:
    study, results = study_results(directory)
    protocol = study["protocol"]
    if protocol.get("feedback_prompt_version") not in {"0.8.0", "0.9.0", "0.10.0", "0.11.0", "0.12.0"}:
        raise ValueError("Frozen concern analysis requires feedback prompt 0.8.0, 0.9.0, 0.10.0, 0.11.0 or 0.12.0.")
    decisions, concerns, issues, measurements = [], [], [], []
    for entry in protocol["entries"]:
        reference = entry["packet"]["reference_inventory_ref"]
        inventory = protocol["reference_inventories"][reference["inventory_id"]]["inventory"]
        central_developmental = [
            c for c in inventory["reference_concerns"]
            if c["centrality"] == "central" and c["concern_type"] == "developmental"
        ]
        eligible_developmental = [
            c for c in central_developmental if c.get("evaluability") != "not_evaluable"
        ]
        actionable_ids = {
            c["concern_id"] for c in inventory["reference_concerns"]
            if c["centrality"] == "central"
            and c["concern_type"] in {"critical", "developmental"}
            and c["reference_support"] in {"supported", "uncertain"}
            and c.get("evaluability") != "not_evaluable"
        }
        positive_ids = {
            c["concern_id"] for c in inventory["reference_concerns"]
            if c["concern_type"] == "positive"
            and c["reference_support"] in {"supported", "uncertain"}
            and c.get("evaluability") != "not_evaluable"
        }
        for repetition in range(1, protocol["judge_repetitions"] + 1):
            base = {
                "document_id": entry["identity"]["document_id"],
                "condition": entry["identity"]["condition"],
                "generation_repetition": entry["generation_repetition"],
                "packet_id": entry["packet"]["packet_id"], "judge_repetition": repetition,
                **reference,
            }
            record = results.get((base["packet_id"], repetition))
            output = record["judgment"]["result"] if record else None
            coverage = {c["concern_id"]: c for c in output["concern_coverage"]} if output else {}
            rate_ids = {
                identifier for identifier in actionable_ids
                if output and coverage[identifier]["coverage"] != "unable_to_judge"
            }
            actionable_counts = {
                status: sum(coverage[c]["coverage"] == status for c in actionable_ids)
                for status in ("missing", "partially_addressed", "addressed")
            } if output else {}
            positive_rate_ids = {
                identifier for identifier in positive_ids
                if output and coverage[identifier]["coverage"] != "unable_to_judge"
            }
            positive_addressed = sum(
                coverage[c]["coverage"] == "addressed" for c in positive_rate_ids
            )
            measurements.append({
                **{key: base[key] for key in FEEDBACK_MEASUREMENT_COLUMNS[:5]},
                "central_actionable_n": len(actionable_ids),
                "central_missing_n": actionable_counts.get("missing"),
                "central_partial_n": actionable_counts.get("partially_addressed"),
                "central_missing_rate": (
                    actionable_counts["missing"] / len(rate_ids) if rate_ids else None
                ),
                "central_full_coverage_rate": (
                    actionable_counts["addressed"] / len(rate_ids) if rate_ids else None
                ),
                "any_major_issue": any(i["severity"] == "major" for i in output["issues"]) if output else None,
                "any_major_contradiction": any(
                    i["severity"] == "major" and i["issue_type"] == "contradiction"
                    for i in output["issues"]
                ) if output else None,
                "overall_severity": output["overall_severity"] if output else "missing",
                "positive_eligible_n": len(positive_ids),
                "positive_addressed_n": positive_addressed if output else None,
                "positive_rate_denominator_n": len(positive_rate_ids) if output else None,
                "positive_recognition_rate": (
                    positive_addressed / len(positive_rate_ids) if positive_rate_ids else None
                ),
                "central_addressed_n": actionable_counts.get("addressed"),
                "central_rate_denominator_n": len(rate_ids) if output else None,
            })
            counts = {
                status: sum(coverage[c["concern_id"]]["coverage"] == status for c in eligible_developmental)
                if output else None
                for status in ("addressed", "partially_addressed", "missing", "unable_to_judge")
            }
            evaluable = len(eligible_developmental) - counts["unable_to_judge"] if output else None
            decisions.append({
                **base, "overall_severity": output["overall_severity"] if output else "missing",
                "rationale": output["rationale"] if output else None,
                "reference_uncertainties": output["reference_uncertainties"] if output else [],
                "central_developmental_count": len(central_developmental),
                "central_developmental_not_evaluable": len(central_developmental) - len(eligible_developmental),
                "central_developmental_partially_evaluable": sum(
                    c.get("evaluability") == "partially_evaluable" for c in central_developmental
                ),
                "not_evaluable_count": sum(
                    c.get("evaluability") == "not_evaluable" for c in inventory["reference_concerns"]
                ),
                "not_applicable_count": (
                    sum(c["coverage"] == "not_applicable" for c in output["concern_coverage"])
                    if output and protocol.get("feedback_prompt_version") in {"0.10.0", "0.11.0", "0.12.0"} else None
                ),
                "central_developmental_not_applicable": (
                    sum(coverage[c["concern_id"]]["coverage"] == "not_applicable" for c in central_developmental)
                    if output and protocol.get("feedback_prompt_version") in {"0.10.0", "0.11.0", "0.12.0"} else None
                ),
                **{f"central_developmental_{status}": count for status, count in counts.items()},
                "central_developmental_evaluable": evaluable,
                "central_developmental_addressed_proportion": (
                    counts["addressed"] / evaluable if evaluable else None
                ),
                "major_issues": sum(i["severity"] == "major" for i in output["issues"]) if output else None,
                "minor_issues": sum(i["severity"] == "minor" for i in output["issues"]) if output else None,
            })
            for concern in inventory["reference_concerns"]:
                assessment = coverage.get(concern["concern_id"])
                concerns.append({
                    **base, **concern,
                    "coverage": (
                        "excluded_not_evaluable" if concern.get("evaluability") == "not_evaluable"
                        and protocol.get("feedback_prompt_version") not in {"0.10.0", "0.11.0", "0.12.0"}
                        else assessment["coverage"] if assessment else "not_run"
                    ),
                    "coverage_rationale": assessment["rationale"] if assessment else None,
                    "generated_references": assessment["generated_references"] if assessment else [],
                    "coverage_evidence_segment_ids": assessment["evidence_segment_ids"] if assessment else [],
                    "central_actionable": concern["concern_id"] in actionable_ids,
                    "positive_eligible": concern["concern_id"] in positive_ids,
                    "in_positive_rate_denominator": concern["concern_id"] in positive_rate_ids,
                    "in_coverage_rate_denominator": concern["concern_id"] in rate_ids,
                    "judgment_available": output is not None,
                    "linked_issue_indices": [
                        index for index, issue in enumerate(output["issues"])
                        if concern["concern_id"] in issue["concern_ids"]
                    ] if output else [],
                })
            if output:
                issues.extend({**base, **issue} for issue in output["issues"])
    issue_table = pd.DataFrame(issues)
    group = ["document_id", "condition", "generation_repetition", "judge_repetition", "issue_type", "severity"]
    issue_counts = (
        issue_table.groupby(group).size().rename("issue_count").reset_index()
        if issues else pd.DataFrame(columns=[*group, "issue_count"])
    )
    return {
        "decisions": pd.DataFrame(decisions), "concern_coverage": pd.DataFrame(concerns),
        "issues": issue_table, "issue_counts": issue_counts,
        "measurements": pd.DataFrame(measurements, columns=FEEDBACK_MEASUREMENT_COLUMNS).astype({
            "central_actionable_n": "Int64", "central_missing_n": "Int64", "central_partial_n": "Int64",
            "central_missing_rate": "Float64", "central_full_coverage_rate": "Float64",
            "any_major_issue": "boolean", "any_major_contradiction": "boolean",
            "positive_eligible_n": "Int64", "positive_addressed_n": "Int64",
            "positive_rate_denominator_n": "Int64", "positive_recognition_rate": "Float64",
            "central_addressed_n": "Int64", "central_rate_denominator_n": "Int64",
        }),
    }


def analyze_alignment_study(directory: str | Path) -> dict[str, pd.DataFrame]:
    study, results = study_results(directory)
    protocol = study["protocol"]
    if protocol.get("assessment_prompt_version") not in {"0.7.0", "0.8.0"}:
        raise ValueError("Alignment analysis requires assessment prompt 0.7.0 or 0.8.0.")
    repetitions = protocol["judge_repetitions"]
    rows = []
    coverage = []
    for entry in protocol["entries"]:
        base = {
            "document_id": entry["identity"]["document_id"],
            "condition": entry["identity"]["condition"],
            "generation_repetition": entry["generation_repetition"],
            "packet_id": entry["packet"]["packet_id"],
        }
        for repetition in range(1, repetitions + 1):
            record = results.get((base["packet_id"], repetition))
            coverage.append({
                **base, "judge_repetition": repetition,
                "status": "missing" if record is None else record["origin"],
            })
            decisions = {} if record is None else {
                d["dimension_id"]: d for d in record["judgment"]["result"]["dimensions"]
            }
            for dimension in entry["packet"]["output"]["dimensions"]:
                decision = decisions.get(dimension["dimension_id"])
                rows.append({
                    **base, "judge_repetition": repetition,
                    "dimension_id": dimension["dimension_id"],
                    "generated_score": dimension["score"],
                    "verdict": "missing" if decision is None else decision["verdict"],
                    "rationale": None if decision is None else decision["rationale"],
                    "half_step_check": None if decision is None else decision["half_step_check"],
                    "explanation_diagnostic": None if decision is None else decision["explanation_diagnostic"],
                })
    decisions = pd.DataFrame(rows)
    keys = ["document_id", "condition", "generation_repetition", "dimension_id"]
    stability, agreement = [], []
    checks = ["lower_anchor_met", "upper_anchor_features_present", "remaining_gap_supported"]
    for values, group in decisions.groupby(keys):
        by_rep = group.set_index("judge_repetition")
        usable = group.loc[~group.verdict.isin(["missing", "unable_to_judge"])]
        complete = len(usable) == repetitions
        verdicts = set(usable.verdict)
        stability.append({
            **dict(zip(keys, values)),
            "expected_repetitions": repetitions, "scored_repetitions": len(usable),
            "missing_judgments": int((group.verdict == "missing").sum()),
            "unable_to_judge": int((group.verdict == "unable_to_judge").sum()),
            "unanimous": len(verdicts) == 1 if complete else None,
            "defensible_to_not_defensible": (
                {"defensible", "not_defensible"}.issubset(verdicts) if complete else None
            ),
            "questionable_transition": (
                "questionable" in verdicts and len(verdicts) > 1 if complete else None
            ),
            "half_step_disagreement": (
                any(len({c[name] for c in usable.half_step_check if c is not None}) > 1
                    for name in checks) if complete else None
            ),
        })
        for left, right in combinations(range(1, repetitions + 1), 2):
            a, b = by_rep.loc[left, "verdict"], by_rep.loc[right, "verdict"]
            comparable = a not in ("missing", "unable_to_judge") and b not in ("missing", "unable_to_judge")
            agreement.append({
                **dict(zip(keys, values)), "first_repetition": left, "second_repetition": right,
                "comparable": comparable, "exact_agreement": a == b if comparable else None,
            })
    return {
        "coverage": pd.DataFrame(coverage),
        "dimension_decisions": decisions,
        "verdict_distribution": decisions.groupby(["condition", "dimension_id", "verdict"]).size()
        .rename("decisions").reset_index(),
        "dimension_stability": pd.DataFrame(stability),
        "pairwise_agreement": pd.DataFrame(agreement),
    }


def analyze_judge_study(directory: str | Path) -> dict[str, pd.DataFrame]:
    study, results = study_results(directory)
    if study["protocol"].get("feedback_prompt_version") in {"0.8.0", "0.9.0", "0.10.0", "0.11.0", "0.12.0"}:
        return analyze_frozen_concern_study(directory)
    if study["protocol"].get("feedback_prompt_version") in {"0.5.0", "0.6.0", "0.7.0"}:
        return analyze_feedback_issue_study(directory)
    if study["protocol"].get("feedback_prompt_version") == "0.4.0":
        return analyze_feedback_alignment_study(directory)
    if study["protocol"].get("assessment_prompt_version") in {"0.7.0", "0.8.0"}:
        return analyze_alignment_study(directory)
    protocol = study["protocol"]
    repetitions = protocol["judge_repetitions"]
    rows = []
    coverage = []
    for entry in protocol["entries"]:
        packet, identity = entry["packet"], entry["identity"]
        base = {
            "cohort": protocol["cohort"], "document_id": identity["document_id"],
            "condition": identity["condition"], "generation_repetition": entry["generation_repetition"],
            "task": packet["task"], "packet_id": packet["packet_id"],
        }
        for repetition in range(1, repetitions + 1):
            record = results.get((packet["packet_id"], repetition))
            coverage.append({**base, "judge_repetition": repetition,
                             "status": "missing" if record is None else record["origin"]})
            if record is not None:
                rows.extend({
                    **base, "judge_repetition": repetition,
                    "criterion_id": rating["criterion_id"], "score": rating["score"],
                    "unable_to_judge": rating["unable_to_judge"],
                } for rating in record["judgment"]["result"]["ratings"])
    columns = list(coverage[0])[:-1] + ["criterion_id", "score", "unable_to_judge"]
    ratings = pd.DataFrame(rows, columns=columns)
    unit_columns = [
        "cohort", "document_id", "condition", "generation_repetition", "task", "criterion_id",
    ]
    unit_rows = []
    expected_units = []
    for entry in protocol["entries"]:
        expected_units.extend({
            "cohort": protocol["cohort"],
            "document_id": entry["identity"]["document_id"],
            "condition": entry["identity"]["condition"],
            "generation_repetition": entry["generation_repetition"],
            "task": entry["packet"]["task"], "criterion_id": criterion["criterion_id"],
        } for criterion in entry["packet"]["criteria"])
    observed_groups = dict(tuple(ratings.groupby(unit_columns)))
    for expected in expected_units:
        values = tuple(expected[name] for name in unit_columns)
        group = observed_groups.get(values, ratings.iloc[:0])
        scored = group.loc[~group["unable_to_judge"] & group["score"].notna(), "score"]
        complete = len(group) == repetitions
        comparable = complete and len(scored) == repetitions
        unit_rows.append({
            **dict(zip(unit_columns, values)), "observed_repetitions": len(group),
            "scored_repetitions": len(scored), "unable_to_judge": int(group["unable_to_judge"].sum()),
            "complete": complete, "comparable": comparable,
            "median": scored.median() if comparable else None,
            "minimum": scored.min() if len(scored) else None,
            "maximum": scored.max() if len(scored) else None,
            "range": scored.max() - scored.min() if len(scored) else None,
            "all_equal": bool(scored.nunique() == 1) if comparable else None,
        })
    units = pd.DataFrame(unit_rows, columns=unit_columns + [
        "observed_repetitions", "scored_repetitions", "unable_to_judge", "complete",
        "comparable", "median", "minimum", "maximum", "range", "all_equal",
    ])
    group_columns = ["cohort", "condition", "task", "criterion_id"]
    summary_rows, agreement_rows = [], []
    for values, group in units.groupby(group_columns):
        eligible = group.loc[group["comparable"]]
        observed = int(group["observed_repetitions"].sum())
        unable = int(group["unable_to_judge"].sum())
        summary_rows.append({
            **dict(zip(group_columns, values)), "expected_packets": len(group),
            "observed_packets": int((group["observed_repetitions"] > 0).sum()),
            "observed_ratings": observed,
            "missing_judgments": int((repetitions - group["observed_repetitions"]).sum()),
            "complete_packets": int(group["complete"].sum()), "comparable_packets": len(eligible),
            "all_equal_packets": int(eligible["all_equal"].sum()),
            "varying_packets": int((eligible["range"] > 0).sum()),
            "extreme_1_to_3_packets": int((eligible["range"] == 2).sum()),
            "all_equal_proportion": eligible["all_equal"].mean() if len(eligible) else None,
            "varying_proportion": (eligible["range"] > 0).mean() if len(eligible) else None,
            "unable_to_judge_ratings": unable,
            "unable_to_judge_proportion": unable / observed if observed else None,
        })
    for values, group in ratings.groupby(group_columns):
        wide = group.pivot(
            index=["document_id", "generation_repetition"], columns="judge_repetition", values="score",
        ).reindex(columns=range(1, repetitions + 1))
        for left, right in combinations(range(1, repetitions + 1), 2):
            pair = wide[[left, right]].dropna()
            agreement_rows.append({
                **dict(zip(group_columns, values)), "first_repetition": left, "second_repetition": right,
                "scored_pairs": len(pair), "exact_agreement_count": int((pair[left] == pair[right]).sum()),
                "exact_agreement": (pair[left] == pair[right]).mean() if len(pair) else None,
            })
    observed_distribution = (
        ratings.loc[ratings["score"].notna()].groupby(group_columns + ["score"]).size()
        .rename("ratings").reset_index()
    )
    distribution = (
        units[group_columns].drop_duplicates()
        .merge(pd.DataFrame({"score": [1, 2, 3]}), how="cross")
        .merge(observed_distribution, on=group_columns + ["score"], how="left")
    )
    distribution["ratings"] = distribution["ratings"].fillna(0).astype(int)
    pairing_index = ["cohort", "document_id", "generation_repetition", "task", "criterion_id"]
    if ratings.empty:
        paired = pd.DataFrame(columns=pairing_index + ["judge_repetition"] + protocol["conditions"])
    else:
        wide = ratings.pivot(
            index=pairing_index + ["judge_repetition"], columns="condition", values="score",
        ).reindex(columns=protocol["conditions"])
        paired = wide.dropna(subset=protocol["conditions"]).reset_index()
    ordering_rows = []
    for _, row in paired.iterrows():
        for left, right in combinations(protocol["conditions"], 2):
            ordering_rows.append({
                **{name: row[name] for name in pairing_index + ["judge_repetition"]},
                "left": left, "right": right,
                "direction": 1 if row[left] > row[right] else -1 if row[left] < row[right] else 0,
            })
    orderings = pd.DataFrame(ordering_rows, columns=pairing_index + [
        "judge_repetition", "left", "right", "direction",
    ])
    ordering_stability = []
    for values, group in orderings.groupby(pairing_index + ["left", "right"]):
        directions = set(group["direction"])
        complete = len(group) == repetitions
        ordering_stability.append({
            **dict(zip(pairing_index + ["left", "right"], values)),
            "observed_repetitions": len(group), "complete": complete,
            "stable_ordering": len(directions) == 1 if complete else None,
            "tie_transition": (0 in directions and len(directions) > 1) if complete else None,
            "strict_reversal": (-1 in directions and 1 in directions) if complete else None,
        })
    ordinal_counts = (
        orderings.groupby(["cohort", "task", "criterion_id", "judge_repetition", "left", "right", "direction"])
        .size().rename("matched_units").reset_index()
    )
    # Secondary medians require all repeats to be scored before matching conditions.
    comparable = units.loc[units["comparable"]]
    if comparable.empty:
        medians = pd.DataFrame(columns=pairing_index + protocol["conditions"])
    else:
        medians = comparable.pivot(
            index=pairing_index, columns="condition", values="median",
        ).reindex(columns=protocol["conditions"]).dropna(subset=protocol["conditions"]).reset_index()
    return {
        "coverage": pd.DataFrame(coverage), "ratings": ratings, "packet_stability": units,
        "criterion_stability": pd.DataFrame(summary_rows),
        "pairwise_agreement": pd.DataFrame(agreement_rows), "rating_distribution": distribution,
        "paired_ratings": paired, "ordinal_counts": ordinal_counts,
        "ordering_stability": pd.DataFrame(ordering_stability),
        "paired_medians": medians,
    }
