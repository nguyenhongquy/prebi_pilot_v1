from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from collections.abc import Iterable
from typing import Any, ClassVar, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reflection_assessment_feedback.experiment_config import load_experiment_config
from reflection_assessment_feedback.prompt_registry import render_prompt, resolve_prompt
from reflection_assessment_feedback.judge_provider import invoke_judge_model
from reflection_assessment_feedback.judge_artifacts import content_hash
from reflection_assessment_feedback.reference_inventory import (
    ReferenceConcernInventory, reference_input_from_packet, validate_inventory,
    EvaluableConcernInventory, parse_inventory,
)

JudgeTask = Literal["assessment_quality", "feedback_quality"]


class StrictJudgeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JudgeCriterionRating(StrictJudgeModel):
    criterion_id: str = Field(min_length=1)
    score: int | None = Field(default=None, ge=1, le=3)
    unable_to_judge: bool
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_judgability(self) -> JudgeCriterionRating:
        if self.unable_to_judge != (self.score is None):
            raise ValueError("Unable-to-judge ratings must have a null score; other ratings need a score.")
        return self


class AssessmentQualityJudgeOutput(StrictJudgeModel):
    ratings: list[JudgeCriterionRating] = Field(min_length=3, max_length=3)
    overall_comment: str

    @model_validator(mode="after")
    def validate_criteria(self) -> AssessmentQualityJudgeOutput:
        _validate_criterion_ids(
            self.ratings,
            {"score_rubric_fit", "evidence_support", "justification_quality"},
        )
        return self


class FeedbackSpanComment(StrictJudgeModel):
    feedback_component: Literal["strengths", "weaknesses", "suggestions"]
    feedback_item_index: int = Field(ge=0)
    start_character: int = Field(ge=0)
    end_character: int = Field(ge=1)
    tags: list[str]
    comment: str = Field(min_length=1)
    linked_reflection_segment_ids: list[str]


class HalfStepCheck(StrictJudgeModel):
    lower_anchor_met: bool
    upper_anchor_features_present: bool
    remaining_gap_supported: bool
    rationale: str = Field(min_length=1)


class DimensionAlignment(StrictJudgeModel):
    dimension_id: Literal["SW", "UA", "HA"]
    generated_score: float = Field(ge=0, le=3)
    verdict: Literal["defensible", "questionable", "not_defensible", "unable_to_judge"]
    rationale: str = Field(min_length=1)
    rubric_requirement: str = Field(min_length=1)
    evidence_segment_ids: list[str]
    half_step_check: HalfStepCheck | None
    explanation_diagnostic: str = Field(min_length=1)


class AssessmentAlignmentJudgeOutput(StrictJudgeModel):
    dimensions: list[DimensionAlignment] = Field(min_length=3, max_length=3)
    overall_comment: str

    @model_validator(mode="after")
    def validate_dimensions(self) -> AssessmentAlignmentJudgeOutput:
        if {d.dimension_id for d in self.dimensions} != {"SW", "UA", "HA"}:
            raise ValueError("Alignment requires exactly SW, UA and HA.")
        for dimension in self.dimensions:
            if dimension.generated_score not in (0, 0.5, 1, 1.5, 2, 2.5, 3):
                raise ValueError("Generated score must be a permitted rubric score.")
            half_step = dimension.generated_score in (0.5, 1.5, 2.5)
            if dimension.verdict == "unable_to_judge":
                if dimension.half_step_check is not None:
                    raise ValueError("Unable-to-judge must not assert half-step checks.")
            elif half_step != (dimension.half_step_check is not None):
                raise ValueError("Scored half-steps require checks; integers must not have them.")
            if dimension.verdict != "unable_to_judge" and not dimension.evidence_segment_ids:
                raise ValueError("Alignment decisions require reflection evidence.")
            check = dimension.half_step_check
            if dimension.verdict == "defensible" and check is not None and not (
                check.lower_anchor_met and check.upper_anchor_features_present
                and check.remaining_gap_supported
            ):
                raise ValueError("A defensible half-step must satisfy every rubric check.")
        return self


def judge_output_model(task: str, prompt_version: str) -> type[StrictJudgeModel]:
    if task == "assessment_quality" and prompt_version in {"0.7.0", "0.8.0"}:
        return AssessmentAlignmentJudgeOutput
    if task == "feedback_quality" and prompt_version == "0.4.0":
        return FeedbackReferenceAlignmentOutput
    if task == "feedback_quality" and prompt_version == "0.5.0":
        return FeedbackIssueJudgeOutput
    if task == "feedback_quality" and prompt_version == "0.6.0":
        return FeedbackFidelityJudgeOutput
    if task == "feedback_quality" and prompt_version == "0.7.0":
        return FeedbackReferenceFirstJudgeOutput
    if task == "feedback_quality" and prompt_version == "0.8.0":
        return FrozenConcernJudgeOutput
    if task == "feedback_quality" and prompt_version == "0.9.0":
        return EvaluableConcernJudgeOutput
    if task == "feedback_quality" and prompt_version == "0.10.0":
        return ApplicableConcernJudgeOutput
    if task == "feedback_quality" and prompt_version == "0.11.0":
        return ReferencePriorityJudgeOutput
    if task == "feedback_quality" and prompt_version == "0.12.0":
        return RefinedReferencePriorityJudgeOutput
    return JUDGE_OUTPUT_MODELS[task]


def validate_alignment_evidence(output: AssessmentAlignmentJudgeOutput, packet: dict[str, Any]) -> None:
    expected = {d["dimension_id"]: d["score"] for d in packet["output"]["dimensions"]}
    if set(expected) != {"SW", "UA", "HA"}:
        raise ValueError("Alignment input requires SW, UA and HA.")
    segments = {s["segment_id"] for s in packet["reflection"]["segments"]}
    for dimension in output.dimensions:
        if dimension.generated_score != expected[dimension.dimension_id]:
            raise ValueError("Alignment changed the generated score.")
        if not set(dimension.evidence_segment_ids).issubset(segments):
            raise ValueError("Alignment cites an unknown reflection segment.")


class FeedbackQualityJudgeOutput(StrictJudgeModel):
    ratings: list[JudgeCriterionRating] = Field(min_length=2, max_length=2)
    overall_comment: str
    span_comments: list[FeedbackSpanComment]

    @model_validator(mode="after")
    def validate_criteria(self) -> FeedbackQualityJudgeOutput:
        _validate_criterion_ids(self.ratings, {"correctness", "developmental_usefulness"})
        return self


class GeneratedFeedbackReference(StrictJudgeModel):
    component: Literal["strengths", "weaknesses", "suggestions"]
    item_index: int = Field(ge=0)
    excerpt: str = Field(min_length=1)


class FeedbackComparison(StrictJudgeModel):
    relationship: Literal[
        "agreement", "central_omission", "contradiction", "compatible_addition",
        "unsupported_addition", "reference_uncertainty",
    ]
    central: bool
    human_excerpt: str | None
    generated_references: list[GeneratedFeedbackReference]
    grounding: Literal["supported", "unsupported", "uncertain", "not_applicable"]
    rationale: str = Field(min_length=1)
    evidence_segment_ids: list[str]

    @model_validator(mode="after")
    def validate_relationship(self):
        if self.relationship not in {"compatible_addition", "unsupported_addition"} and not self.human_excerpt:
            raise ValueError("Reference comparisons require a human feedback excerpt.")
        if self.relationship not in {"central_omission", "reference_uncertainty"} and not self.generated_references:
            raise ValueError("This relationship requires generated feedback references.")
        if self.relationship == "central_omission" and (not self.central or self.generated_references):
            raise ValueError("Central omissions must be central and have no matching generated item.")
        return self


class FeedbackReferenceAlignmentOutput(StrictJudgeModel):
    verdict: Literal["aligned", "partially_aligned", "misaligned", "unable_to_judge"]
    rationale: str = Field(min_length=1)
    comparisons: list[FeedbackComparison]
    overall_comment: str

    @model_validator(mode="after")
    def validate_verdict(self):
        if self.verdict != "unable_to_judge" and not self.comparisons:
            raise ValueError("Alignment judgments require concrete comparisons.")
        issues = [
            c for c in self.comparisons
            if c.central and (
                c.relationship == "central_omission"
                or c.relationship in {"contradiction", "unsupported_addition"} and c.grounding == "unsupported"
            )
        ]
        if self.verdict == "aligned" and issues:
            raise ValueError("Aligned verdict cannot contain supported central alignment gaps.")
        if self.verdict == "misaligned" and not any(
            c.relationship in {"contradiction", "unsupported_addition"}
            and c.central and c.grounding == "unsupported" for c in self.comparisons
        ):
            raise ValueError("Misalignment requires a central, grounded substantive conflict.")
        return self


class ReferenceConcern(StrictJudgeModel):
    concern_id: str = Field(min_length=1)
    human_excerpt: str = Field(min_length=1)
    centrality_rationale: str = Field(min_length=1)
    reference_support: Literal["supported", "uncertain", "unsupported"]
    coverage: Literal["addressed", "partially_addressed", "missing", "unable_to_judge"]
    generated_references: list[GeneratedFeedbackReference]
    evidence_segment_ids: list[str]
    rationale: str = Field(min_length=1)


class FeedbackIssue(StrictJudgeModel):
    reference_matching: ClassVar[bool] = True
    issue_type: Literal[
        "central_omission", "contradiction", "unsupported_claim", "misleading_emphasis",
        "local_imprecision", "secondary_omission",
    ]
    severity: Literal["major", "minor"]
    concern_ids: list[str]
    human_excerpt: str | None
    generated_references: list[GeneratedFeedbackReference]
    evidence_segment_ids: list[str]
    rationale: str = Field(min_length=1)
    consequence: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_issue(self):
        if self.reference_matching and self.issue_type in {"central_omission", "contradiction"} and self.severity != "major":
            raise ValueError("Central omissions and substantive contradictions are major.")
        if self.reference_matching and self.issue_type == "central_omission" and not self.concern_ids:
            raise ValueError("Central omission requires a previously identified concern.")
        if self.reference_matching and self.issue_type in {"central_omission", "secondary_omission", "contradiction"} and not self.human_excerpt:
            raise ValueError("Reference-related issues require human excerpts.")
        if self.issue_type not in {"central_omission", "secondary_omission"} and not self.generated_references:
            raise ValueError("Generated assertions require exact item references.")
        if self.issue_type == "secondary_omission" and self.severity != "minor":
            raise ValueError("Secondary omissions are minor.")
        return self


class FeedbackFidelityIssue(FeedbackIssue):
    reference_matching: ClassVar[bool] = False

    @model_validator(mode="after")
    def validate_local_severity(self):
        if self.issue_type == "local_imprecision" and self.severity != "minor":
            raise ValueError("Local imprecision is minor.")
        return self


class FeedbackFidelityUncertainty(FeedbackComparison):
    @model_validator(mode="after")
    def validate_relationship(self):
        if self.relationship != "reference_uncertainty":
            raise ValueError("Uncertainties must not be disguised generated errors.")
        return self


class FeedbackIssueJudgeOutput(StrictJudgeModel):
    reference_matching: ClassVar[bool] = True
    reference_concerns: list[ReferenceConcern]
    issues: list[FeedbackIssue]
    overall_severity: Literal["major", "minor", "no_error", "unable_to_judge"]
    rationale: str = Field(min_length=1)
    reference_uncertainties: list[FeedbackComparison]

    @model_validator(mode="after")
    def validate_aggregation(self):
        concerns = {c.concern_id: c for c in self.reference_concerns}
        if len(concerns) != len(self.reference_concerns):
            raise ValueError("Concern IDs must be unique.")
        for issue in self.issues:
            if not set(issue.concern_ids).issubset(concerns):
                raise ValueError("Issue references an unknown concern.")
            if issue.issue_type == "central_omission" and not all(
                concerns[c].reference_support == "supported"
                and concerns[c].coverage in {"missing", "partially_addressed"}
                for c in issue.concern_ids
            ):
                raise ValueError("Central omissions require supported, incompletely addressed concerns.")
        for c in self.reference_concerns:
            if self.reference_matching and c.reference_support == "supported" and c.coverage == "missing" and not any(
                i.issue_type == "central_omission" and c.concern_id in i.concern_ids for i in self.issues
            ):
                raise ValueError("A missing supported central concern must be recorded as major.")
        if any(c.relationship != "reference_uncertainty" for c in self.reference_uncertainties):
            raise ValueError("Reference uncertainties must not be disguised generated errors.")
        expected = "major" if any(i.severity == "major" for i in self.issues) else "minor" if self.issues else "no_error"
        if self.overall_severity != "unable_to_judge" and self.overall_severity != expected:
            raise ValueError("Overall severity must equal the highest issue severity.")
        if self.reference_matching and self.overall_severity == "no_error" and any(
            c.coverage == "unable_to_judge" for c in self.reference_concerns
        ):
            raise ValueError("Unresolved coverage cannot be reported as no error.")
        return self


class FeedbackFidelityJudgeOutput(FeedbackIssueJudgeOutput):
    reference_matching: ClassVar[bool] = False
    issues: list[FeedbackFidelityIssue]
    reference_uncertainties: list[FeedbackFidelityUncertainty]

    @model_validator(mode="after")
    def validate_decisive_uncertainty(self):
        if self.overall_severity == "unable_to_judge" and not self.reference_uncertainties:
            raise ValueError("Unable-to-judge requires a documented decisive uncertainty.")
        if self.overall_severity == "unable_to_judge" and any(
            issue.severity == "major" for issue in self.issues
        ):
            raise ValueError("An established major issue determines overall severity.")
        return self


class FeedbackReferenceFirstJudgeOutput(FeedbackFidelityJudgeOutput):
    @model_validator(mode="after")
    def validate_reference_first_issues(self):
        concerns = {c.concern_id: c for c in self.reference_concerns}
        for issue in self.issues:
            if issue.issue_type in {"central_omission", "secondary_omission", "contradiction"}:
                if not issue.concern_ids or not issue.human_excerpt:
                    raise ValueError("Reference-first omissions and contradictions require reference concerns and excerpts.")
                if any(concerns[c].reference_support != "supported" for c in issue.concern_ids):
                    raise ValueError("Reference-first issues require supported reference concerns.")
        return self


class ConcernCoverage(StrictJudgeModel):
    concern_id: str = Field(min_length=1)
    coverage: Literal["addressed", "partially_addressed", "missing", "unable_to_judge"]
    generated_references: list[GeneratedFeedbackReference]
    evidence_segment_ids: list[str]
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_coverage_evidence(self):
        if self.coverage in {"addressed", "partially_addressed"} and not self.generated_references:
            raise ValueError("Addressed concerns require exact generated feedback references.")
        return self


class FrozenConcernJudgeOutput[
    VersionT: str = Literal["2.0.0"],
    CoverageT: ConcernCoverage = ConcernCoverage,
    IssueT: FeedbackFidelityIssue = FeedbackFidelityIssue,
](StrictJudgeModel):
    schema_version: VersionT
    concern_coverage: list[CoverageT]
    issues: list[IssueT]
    overall_severity: Literal["major", "minor", "no_error", "unable_to_judge"]
    rationale: str = Field(min_length=1)
    reference_uncertainties: list[FeedbackFidelityUncertainty]

    @model_validator(mode="after")
    def validate_overall(self):
        identifiers = [c.concern_id for c in self.concern_coverage]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Every supplied concern needs exactly one coverage result.")
        expected = "major" if any(i.severity == "major" for i in self.issues) else "minor" if self.issues else "no_error"
        if self.overall_severity == "unable_to_judge":
            if not self.reference_uncertainties or expected == "major":
                raise ValueError("Unable-to-judge needs decisive uncertainty and no established major issue.")
        elif self.overall_severity != expected:
            raise ValueError("Overall severity must equal the highest issue severity.")
        return self


class EvaluableConcernJudgeOutput(FrozenConcernJudgeOutput[Literal["2.1.0"]]):
    pass


class ApplicableConcernCoverage(ConcernCoverage):
    coverage: Literal["addressed", "partially_addressed", "missing", "unable_to_judge", "not_applicable"]

    @model_validator(mode="after")
    def validate_not_applicable(self):
        if self.coverage == "not_applicable" and (self.generated_references or self.evidence_segment_ids):
            raise ValueError("Not-applicable coverage must not assert generated coverage or reflection evidence.")
        return self


class ApplicableConcernJudgeOutput[
    VersionT: str = Literal["2.2.0"],
    IssueT: FeedbackFidelityIssue = FeedbackFidelityIssue,
](FrozenConcernJudgeOutput[VersionT, ApplicableConcernCoverage, IssueT]):
    pass


class ReferencePriorityIssue(FeedbackFidelityIssue):
    severity_basis: Literal["documented_reference_deviation", "independent_generated_error"]


class ReferencePriorityJudgeOutput(
    ApplicableConcernJudgeOutput[Literal["2.3.0"], ReferencePriorityIssue],
):
    pass


class RefinedReferencePriorityJudgeOutput(ReferencePriorityJudgeOutput):
    pass


def validate_frozen_concern_output(output: FrozenConcernJudgeOutput, packet: dict) -> None:
    inventory = parse_inventory(packet["reference_inventory"])
    if isinstance(output, (EvaluableConcernJudgeOutput, ApplicableConcernJudgeOutput)) and not isinstance(inventory, EvaluableConcernInventory):
        raise ValueError("Evaluability-aware judging requires inventory schema 1.1.0.")
    concerns = {
        c.concern_id: c for c in inventory.reference_concerns
        if not isinstance(output, EvaluableConcernJudgeOutput)
        or c.evaluability != "not_evaluable"
    }
    coverage = {c.concern_id: c for c in output.concern_coverage}
    if coverage.keys() != concerns.keys():
        raise ValueError("Coverage must match every supplied inventory concern exactly once.")
    if isinstance(output, ApplicableConcernJudgeOutput):
        for identifier, concern in concerns.items():
            if (coverage[identifier].coverage == "not_applicable") != (
                concern.evaluability == "not_evaluable"
            ):
                raise ValueError("Not-applicable coverage must match frozen not_evaluable concerns exactly.")
    for issue in output.issues:
        if not set(issue.concern_ids).issubset(concerns):
            raise ValueError("Issue references an unknown inventory concern.")
        if isinstance(output, ApplicableConcernJudgeOutput) and any(
            concerns[c].evaluability == "not_evaluable" for c in issue.concern_ids
        ):
            raise ValueError("Non-evaluable reference concerns cannot generate issues.")
        independent_contradiction = (
            isinstance(output, ApplicableConcernJudgeOutput)
            and issue.issue_type == "contradiction" and not issue.concern_ids
        )
        if independent_contradiction and not issue.evidence_segment_ids:
            raise ValueError("Independent contradictions require reflection evidence.")
        if issue.issue_type in {"central_omission", "secondary_omission", "contradiction"} and not independent_contradiction:
            if not issue.concern_ids or not issue.human_excerpt:
                raise ValueError("Reference-first issues require inventory concerns and human excerpts.")
            if not isinstance(output, ReferencePriorityJudgeOutput) and any(
                concerns[c].reference_support != "supported" for c in issue.concern_ids
            ):
                raise ValueError("Reference-first issues require supported inventory concerns.")
        if isinstance(issue, ReferencePriorityIssue):
            if issue.concern_ids:
                if issue.severity_basis != "documented_reference_deviation":
                    raise ValueError("Reference issues must explicitly identify documented reference deviation.")
                if any(concerns[c].reference_support == "unsupported" for c in issue.concern_ids):
                    raise ValueError("Unsupported reference concerns cannot justify reference-deviation issues.")
                if not isinstance(output, RefinedReferencePriorityJudgeOutput) and any(
                    concerns[c].reference_support == "uncertain" and concerns[c].centrality != "central"
                    for c in issue.concern_ids
                ):
                    raise ValueError("Uncertain reference-deviation issues require central concerns.")
                if isinstance(output, RefinedReferencePriorityJudgeOutput) and issue.severity == "major" and not any(
                    concerns[c].centrality == "central" for c in issue.concern_ids
                ):
                    raise ValueError("Major reference-deviation issues require a central concern.")
            elif issue.severity_basis != "independent_generated_error":
                raise ValueError("Independent generated errors must identify their independent severity basis.")
        if issue.issue_type in {"central_omission", "secondary_omission"} and any(
            coverage[c].coverage not in {"missing", "partially_addressed"} for c in issue.concern_ids
        ):
            raise ValueError("Omissions require incompletely addressed inventory concerns.")
        if issue.issue_type == "central_omission" and any(
            concerns[c].centrality != "central" for c in issue.concern_ids
        ):
            raise ValueError("Central omission requires central inventory concerns.")
    _validate_feedback_citations(output.issues, packet)
    _validate_feedback_citations(output.reference_uncertainties, packet)
    _validate_feedback_citations(output.concern_coverage, packet)


def _validate_feedback_citations(
    comparisons: Iterable[FeedbackComparison | ReferenceConcern | FeedbackIssue | ConcernCoverage],
    packet: dict[str, Any],
) -> None:
    def contains_quote(text: str, excerpt: str) -> bool:
        return bool(excerpt.strip()) and " ".join(excerpt.split()) in " ".join(text.split())

    segments = {s["segment_id"] for s in packet["reflection"]["segments"]}
    for comparison in comparisons:
        if not isinstance(comparison, ConcernCoverage) and comparison.human_excerpt is not None and (
            not contains_quote(packet["reference_teacher_feedback"], comparison.human_excerpt)
        ):
            raise ValueError("Human excerpt is not present in the reference feedback.")
        if not set(comparison.evidence_segment_ids).issubset(segments):
            raise ValueError("Feedback comparison cites an unknown reflection segment.")
        for reference in comparison.generated_references:
            items = packet["output"].get(reference.component, [])
            if reference.item_index >= len(items) or not contains_quote(
                items[reference.item_index]["text"], reference.excerpt,
            ):
                raise ValueError("Generated excerpt does not match the referenced feedback item.")


def validate_feedback_alignment(output: FeedbackReferenceAlignmentOutput, packet: dict[str, Any]) -> None:
    _validate_feedback_citations(output.comparisons, packet)


def validate_feedback_issues(output: FeedbackIssueJudgeOutput, packet: dict[str, Any]) -> None:
    _validate_feedback_citations(output.reference_concerns, packet)
    _validate_feedback_citations(output.issues, packet)
    _validate_feedback_citations(output.reference_uncertainties, packet)


JUDGE_OUTPUT_MODELS: dict[str, type[StrictJudgeModel]] = {
    "assessment_quality": AssessmentQualityJudgeOutput,
    "feedback_quality": FeedbackQualityJudgeOutput,
}


def _validate_criterion_ids(ratings: list[JudgeCriterionRating], expected: set[str]) -> None:
    actual = [rating.criterion_id for rating in ratings]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(f"Ratings must contain exactly these criteria: {sorted(expected)}.")


def _canonical_sha256(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_judge_input(
    packet: dict[str, Any], *, assessment_prompt_version: str | None = None,
    feedback_prompt_version: str | None = None,
) -> dict[str, Any]:
    task = packet.get("task")
    if task not in JUDGE_OUTPUT_MODELS:
        raise ValueError("Only assessment-quality and feedback-quality packets can be judged.")
    if feedback_prompt_version is not None and task != "feedback_quality":
        raise ValueError("Feedback prompt version requires a feedback-quality packet.")
    if assessment_prompt_version == "0.8.0":
        if task != "assessment_quality":
            raise ValueError("Reference-blind judging requires an assessment-quality packet.")
        required_blind = {"reflection", "rubric", "output"}
        if not required_blind.issubset(packet):
            raise ValueError("Reference-blind judge requires reflection, rubric and output.")
        return {name: packet[name] for name in ("reflection", "rubric", "output")}
    required = {"reflection", "rubric", "output", "reference_teacher_feedback"}
    if task == "assessment_quality":
        required.add("gold_analysis_summary")
    if not required.issubset(packet):
        raise ValueError(
            "Judge packet is missing reflection, rubric, generated output, teacher reference feedback, "
            "or the assessment gold-analysis summary."
        )
    reference_feedback = packet["reference_teacher_feedback"]
    if not isinstance(reference_feedback, str) or not reference_feedback.strip():
        raise ValueError("Teacher reference feedback must be a non-empty string.")
    judge_input = {
        name: packet[name]
        for name in ("reflection", "rubric", "output", "reference_teacher_feedback")
    }
    if task == "assessment_quality":
        summary = packet["gold_analysis_summary"]
        if not isinstance(summary, dict) or summary.get("analysis_source") != "provisional_gold_candidate_annotations":
            raise ValueError("Assessment judge requires a provisional gold-analysis summary.")
        judge_input["gold_analysis_summary"] = summary
    if task == "feedback_quality" and feedback_prompt_version in {"0.8.0", "0.9.0", "0.10.0", "0.11.0", "0.12.0"}:
        if "reference_inventory" not in packet or "reference_inventory_ref" not in packet:
            raise ValueError("Frozen-concern judge requires an approved reference inventory.")
        inventory = parse_inventory(packet["reference_inventory"])
        if feedback_prompt_version in {"0.9.0", "0.10.0", "0.11.0", "0.12.0"} and not isinstance(inventory, EvaluableConcernInventory):
            raise ValueError("Evaluability-aware judging requires inventory schema 1.1.0.")
        if feedback_prompt_version == "0.8.0" and isinstance(inventory, EvaluableConcernInventory):
            raise ValueError("Inventory schema 1.1.0 requires feedback prompt 0.9.0.")
        reference = packet["reference_inventory_ref"]
        if (
            content_hash(packet["reference_inventory"]) != reference["inventory_sha256"]
            or inventory.reflection_id != reference["reflection_id"]
        ):
            raise ValueError("Reference inventory hash or reflection identity mismatch.")
        validate_inventory(inventory, reference_input_from_packet(packet, inventory.reflection_id))
        judge_input["reference_inventory"] = inventory.model_dump(mode="json")
        if feedback_prompt_version == "0.9.0":
            judge_input["reference_inventory"]["reference_concerns"] = [
                c for c in judge_input["reference_inventory"]["reference_concerns"]
                if c["evaluability"] != "not_evaluable"
            ]
    return judge_input


def judge_input_sha256(
    packet: dict[str, Any], *, assessment_prompt_version: str | None = None,
    feedback_prompt_version: str | None = None,
) -> str:
    return _canonical_sha256(build_judge_input(
        packet, assessment_prompt_version=assessment_prompt_version,
        feedback_prompt_version=feedback_prompt_version,
    ))


def _validate_span_comments(
    output: FeedbackQualityJudgeOutput,
    judge_input: dict[str, Any],
) -> None:
    feedback = judge_input["output"]
    segment_ids = {
        segment["segment_id"]
        for segment in judge_input["reflection"].get("segments", [])
        if "segment_id" in segment
    }
    for comment in output.span_comments:
        items = feedback.get(comment.feedback_component, [])
        if comment.feedback_item_index >= len(items):
            raise ValueError("A span comment references a missing feedback item.")
        text = items[comment.feedback_item_index].get("text", "")
        if comment.start_character >= comment.end_character or comment.end_character > len(text):
            raise ValueError("A span comment has invalid offsets for its feedback item.")
        if not set(comment.linked_reflection_segment_ids).issubset(segment_ids):
            raise ValueError("A span comment references an unknown reflection segment.")


def run_quality_judge(
    *,
    model: Any,
    packet: dict[str, Any],
    provider: str,
    model_name: str,
    enable_langsmith_tracing: bool = False,
    assessment_prompt_version: str | None = None,
    feedback_prompt_version: str | None = None,
) -> dict[str, Any]:
    task = packet.get("task")
    if task not in JUDGE_OUTPUT_MODELS:
        raise ValueError("Only assessment-quality and feedback-quality packets can be judged.")
    if not provider.strip() or not model_name.strip():
        raise ValueError("Judge provider and model name must be configured.")
    if assessment_prompt_version is not None and task != "assessment_quality":
        raise ValueError("Assessment prompt override requires an assessment-quality packet.")
    if feedback_prompt_version is not None and task != "feedback_quality":
        raise ValueError("Feedback prompt override requires a feedback-quality packet.")

    prompt_config = load_experiment_config()["prompts"]
    prompt_artifact = resolve_prompt(
        prompt_config[f"{task}_llm_id"],
        assessment_prompt_version if assessment_prompt_version is not None
        else feedback_prompt_version if feedback_prompt_version is not None
        else prompt_config[f"{task}_llm_version"],
    )
    if prompt_artifact.task != task or prompt_artifact.modality != "llm_prompt":
        raise ValueError("Configured judge prompt is not registered for this task.")
    if prompt_artifact.status != "development_draft":
        raise ValueError("This runner is restricted to development-draft judge prompts.")

    judge_input = build_judge_input(
        packet,
        assessment_prompt_version=prompt_artifact.version if task == "assessment_quality" else None,
        feedback_prompt_version=prompt_artifact.version if task == "feedback_quality" else None,
    )
    packet_json = json.dumps(judge_input, ensure_ascii=False, sort_keys=True, indent=2)
    rendered_prompt, rendered_prompt_sha256 = render_prompt(
        prompt_artifact,
        packet_json=packet_json,
    )
    packet_sha256 = _canonical_sha256(judge_input)
    input_run_id = str(uuid4())

    output_model = judge_output_model(task, prompt_artifact.version)
    trace_metadata = {
        "task": task,
        "run_id": input_run_id,
        "prompt_id": prompt_artifact.artifact_id,
        "prompt_version": prompt_artifact.version,
        "prompt_artifact_sha256": prompt_artifact.sha256,
        "rendered_prompt_sha256": rendered_prompt_sha256,
    }
    if task == "feedback_quality" and prompt_artifact.version in {"0.8.0", "0.9.0", "0.10.0", "0.11.0", "0.12.0"}:
        trace_metadata["reference_inventory_ref"] = packet["reference_inventory_ref"]
    raw_output = invoke_judge_model(
        model=model, schema=output_model, prompt=rendered_prompt, metadata=trace_metadata,
        enable_langsmith_tracing=enable_langsmith_tracing,
    )

    validated_output = (
        raw_output
        if isinstance(raw_output, output_model)
        else output_model.model_validate(raw_output)
    )
    if isinstance(validated_output, FeedbackQualityJudgeOutput):
        _validate_span_comments(validated_output, judge_input)
    if isinstance(validated_output, AssessmentAlignmentJudgeOutput):
        validate_alignment_evidence(validated_output, packet)
    if isinstance(validated_output, FeedbackReferenceAlignmentOutput):
        validate_feedback_alignment(validated_output, packet)
    if isinstance(validated_output, FeedbackIssueJudgeOutput):
        validate_feedback_issues(validated_output, packet)
    if isinstance(validated_output, FrozenConcernJudgeOutput):
        validate_frozen_concern_output(validated_output, packet)

    result = {
        "schema_version": "1.0.0",
        "record_type": "llm_judge_quality_rating",
        "run_id": input_run_id,
        "packet_id": packet.get("packet_id"),
        "task": task,
        "status": "development_judged",
        "evaluator_type": "llm_judge",
        "judge_provider": provider,
        "judge_model": model_name,
        "judge_parameters": {
            name: value
            for name in ("top_p", "max_tokens", "seed")
            if (value := getattr(model, name, None)) is not None
        },
        "prompt_id": prompt_artifact.artifact_id,
        "prompt_version": prompt_artifact.version,
        "prompt_artifact_sha256": prompt_artifact.sha256,
        "rendered_prompt_sha256": rendered_prompt_sha256,
        "input_packet_sha256": packet_sha256,
        "teacher_reference_sha256": hashlib.sha256(
            judge_input["reference_teacher_feedback"].encode("utf-8")
        ).hexdigest() if "reference_teacher_feedback" in judge_input else None,
        "rubric_version": packet.get("rubric_version"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "result": validated_output.model_dump(mode="json"),
    }
    if isinstance(validated_output, FrozenConcernJudgeOutput):
        result["schema_version"] = validated_output.schema_version
        result["reference_inventory_ref"] = packet["reference_inventory_ref"]
    if task == "assessment_quality" and "gold_analysis_summary" in judge_input:
        result["gold_analysis_summary_sha256"] = _canonical_sha256(
            judge_input["gold_analysis_summary"]
        )
    elif task == "assessment_quality":
        result["gold_analysis_summary_sha256"] = None
    return result