from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reflection_assessment_feedback.judge_study import (
    _private_directory, _protected, _write_once, content_hash, load_judge_study,
)


class ReviewFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    issue: Literal["theory_required", "all_facets_required", "cross_dimension_chain",
                   "extra_completeness", "half_point_explanation", "other"]
    classification: Literal["supported_requirement", "development_suggestion",
                            "unsupported_scoring_requirement", "ambiguous"]
    assessment_excerpt: str = Field(min_length=1)
    rubric_clause: str = Field(min_length=1)
    affects_score: bool | None
    evidence_segment_ids: list[str]
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def nonblank_support(self):
        if not all(text.strip() for text in (
            self.assessment_excerpt, self.rubric_clause, self.rationale,
        )):
            raise ValueError("Findings require nonblank excerpts and rationale.")
        return self


class DimensionReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    status: Literal["pending", "reviewed"] = "pending"
    reviewer: str = ""
    findings: list[ReviewFinding] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def completed_review(self):
        if self.status == "reviewed" and (not self.reviewer.strip() or not self.notes.strip()):
            raise ValueError("Reviewed cases require a reviewer and explicit rationale, including no-issue cases.")
        return self


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def rubric_review_cases(directory: str | Path) -> list[dict]:
    study = load_judge_study(directory)
    cases = []
    for entry in study["protocol"]["entries"]:
        packet = entry["packet"]
        if packet["task"] != "assessment_quality":
            continue
        source = json.loads(study["protocol"]["generation_sources"][entry["identity"]["source_artifact_sha256"]])
        if "run" in source:
            run = source["run"]
        else:
            run = next(
                pair[entry["identity"]["condition"]]
                for pair in source["pairs"]
                if pair["repetition"] == entry["generation_repetition"]
            )
        synthesis = run.get("output", run).get("evidence_synthesis")
        for dimension in packet["output"]["dimensions"]:
            cases.append({
                "case_id": f"{packet['packet_id']}:{dimension['dimension_id']}",
                "study_id": study["study_id"], **entry["identity"],
                "dimension_id": dimension["dimension_id"], "assessment": dimension,
                "evidence_synthesis": synthesis,
                "reflection": packet["reflection"], "rubric": packet["rubric"],
            })
    return cases


def create_rubric_review(directory: str | Path) -> Path:
    directory = _protected(directory)
    study = load_judge_study(directory)
    review_directory = directory / "rubric-review"
    _private_directory(review_directory, directory)
    path = review_directory / "worksheet.json"
    if path.exists():
        validate_rubric_review(directory, json.loads(path.read_text()))
    else:
        _write_once(path, {
            "study_id": study["study_id"],
            "reviews": [DimensionReview(case_id=case["case_id"]).model_dump()
                        for case in rubric_review_cases(directory)],
        })
    return path


def validate_rubric_review(directory: str | Path, worksheet: dict) -> list[DimensionReview]:
    study = load_judge_study(directory)
    if set(worksheet) != {"study_id", "reviews"} or worksheet["study_id"] != study["study_id"]:
        raise ValueError("Review worksheet must belong to this frozen study.")
    cases = {case["case_id"]: case for case in rubric_review_cases(directory)}
    reviews = [DimensionReview.model_validate(item) for item in worksheet["reviews"]]
    if len(reviews) != len(cases) or {r.case_id for r in reviews} != set(cases):
        raise ValueError("Review cases must be unique and cover every frozen assessment dimension.")
    for review in reviews:
        case = cases[review.case_id]
        segments = {str(s["segment_id"]) for s in case["reflection"]["segments"]}
        for finding in review.findings:
            if not any(finding.assessment_excerpt in text for text in _strings(case["assessment"])):
                raise ValueError("Review excerpt does not resolve to the frozen assessment.")
            if not any(finding.rubric_clause in text for text in _strings(case["rubric"])):
                raise ValueError("Rubric clause does not resolve to the frozen rubric.")
            if not set(finding.evidence_segment_ids).issubset(segments):
                raise ValueError("Review evidence reference does not resolve to the original reflection.")
    return reviews


def save_rubric_review_revision(directory: str | Path, worksheet: dict) -> Path:
    directory = _protected(directory)
    reviews = validate_rubric_review(directory, worksheet)
    content = {"study_id": worksheet["study_id"], "reviews": [r.model_dump() for r in reviews]}
    review_directory = directory / "rubric-review"
    _private_directory(review_directory, directory)
    path = review_directory / f"review-{content_hash(content)}.json"
    if path.exists():
        if json.loads(path.read_text()) != content:
            raise ValueError("Review revision content mismatch.")
    else:
        _write_once(path, content)
    return path
