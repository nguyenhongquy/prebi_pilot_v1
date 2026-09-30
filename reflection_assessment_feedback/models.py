from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

Condition = Literal["G1", "G2", "G3"]
AnalysisSource = Literal["predicted", "human"]
Scope = Literal["in_scope", "out_of_scope"]
ComponentLabel = Literal["N", "0", "1", "2", "3"]
DIMENSION_IDS = ("SW", "UA", "HA")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReflectionSegment(StrictModel):
    segment_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    order: int = Field(ge=0)


class ReflectionDocument(StrictModel):
    document_id: str = Field(min_length=1)
    segments: list[ReflectionSegment] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_segments(self) -> ReflectionDocument:
        ids = [segment.segment_id for segment in self.segments]
        orders = [segment.order for segment in self.segments]
        if len(ids) != len(set(ids)):
            raise ValueError("A document must not contain duplicate segment IDs.")
        if len(orders) != len(set(orders)):
            raise ValueError("A document must not contain duplicate segment orders.")
        if orders != sorted(orders):
            raise ValueError("Segments must be provided in source order.")
        return self


class RubricDimension(StrictModel):
    dimension_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    bands: dict[str, str] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def validate_bands(self) -> RubricDimension:
        if set(self.bands) != {"0", "1", "2", "3"}:
            raise ValueError("Each rubric dimension must define Bands 0, 1, 2, and 3.")
        return self


class Rubric(StrictModel):
    rubric_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    glossary: dict[str, str] = Field(min_length=1)
    dimensions: list[RubricDimension] = Field(min_length=3)

    @model_validator(mode="after")
    def validate_dimensions(self) -> Rubric:
        ids = [dimension.dimension_id for dimension in self.dimensions]
        if len(ids) != len(set(ids)):
            raise ValueError("Rubric dimension IDs must be unique.")
        if set(ids) != set(DIMENSION_IDS):
            raise ValueError(f"The experiment rubric must define {DIMENSION_IDS}.")
        return self


class SegmentAnalysis(StrictModel):
    segment_id: str = Field(min_length=1)
    scope: Scope
    component_bands: dict[str, ComponentLabel] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_scope_labels(self) -> SegmentAnalysis:
        if set(self.component_bands) != set(DIMENSION_IDS):
            raise ValueError(f"Analysis must label exactly {DIMENSION_IDS}.")
        expected = "N" if self.scope == "out_of_scope" else None
        if expected and any(value != expected for value in self.component_bands.values()):
            raise ValueError("Out-of-scope segments must have N for every component.")
        if self.scope == "in_scope" and any(
            value == "N" for value in self.component_bands.values()
        ):
            raise ValueError("In-scope segments must use Bands 0-3, not N.")
        return self


class DimensionAnalysisSummary(StrictModel):
    dimension_id: str = Field(min_length=1)
    in_scope_segment_count: int = Field(ge=1)
    positive_segment_count: int = Field(ge=0)
    positive_segment_percentage: float = Field(ge=0, le=100)
    positive_band_mean: float | None = Field(default=None, ge=1, le=3)
    positive_band_percentages: dict[str, float] | None = None

    @model_validator(mode="after")
    def validate_band_percentages(self) -> DimensionAnalysisSummary:
        if self.positive_band_percentages is not None:
            if set(self.positive_band_percentages) != {"1", "2", "3"}:
                raise ValueError("Positive band percentages must define Bands 1, 2, and 3.")
            if any(
                not 0 <= percentage <= 100
                for percentage in self.positive_band_percentages.values()
            ):
                raise ValueError("Positive band percentages must be between 0 and 100.")
        return self


class IntermediateAnalysis(StrictModel):
    source: AnalysisSource
    segments: list[SegmentAnalysis] = Field(min_length=1)

    @computed_field
    @property
    def summary(self) -> list[DimensionAnalysisSummary] | None:
        in_scope = [segment for segment in self.segments if segment.scope == "in_scope"]
        if not in_scope:
            return None

        summaries = []
        for dimension_id in DIMENSION_IDS:
            scores = [int(segment.component_bands[dimension_id]) for segment in in_scope]
            count = len(scores)
            positive_scores = [score for score in scores if score > 0]
            summaries.append(
                DimensionAnalysisSummary(
                    dimension_id=dimension_id,
                    in_scope_segment_count=count,
                    positive_segment_count=len(positive_scores),
                    positive_segment_percentage=round(len(positive_scores) / count * 100, 2),
                    positive_band_mean=(
                        round(sum(positive_scores) / len(positive_scores), 2)
                        if positive_scores else None
                    ),
                    positive_band_percentages=(
                        {
                            str(band): round(
                                positive_scores.count(band) / len(positive_scores) * 100, 2
                            )
                            for band in range(1, 4)
                        }
                        if positive_scores else None
                    ),
                )
            )
        return summaries


class GenerationRequest(StrictModel):
    condition: Condition
    document: ReflectionDocument
    rubric: Rubric
    analysis: IntermediateAnalysis | None = None

    @model_validator(mode="after")
    def validate_analysis(self) -> GenerationRequest:
        required_source = {"G2": "predicted", "G3": "human"}.get(self.condition)
        if required_source is None and self.analysis is not None:
            raise ValueError("G1 must not receive segment analysis.")
        if required_source is not None and self.analysis is None:
            raise ValueError(f"{self.condition} requires {required_source} analysis.")
        if self.analysis is None:
            return self
        if self.analysis.source != required_source:
            raise ValueError(
                f"{self.condition} requires analysis source {required_source!r}."
            )
        document_ids = {segment.segment_id for segment in self.document.segments}
        analysis_ids = [segment.segment_id for segment in self.analysis.segments]
        if len(analysis_ids) != len(set(analysis_ids)):
            raise ValueError("Analysis must not contain duplicate segment IDs.")
        if set(analysis_ids) != document_ids:
            raise ValueError("Analysis must cover each document segment exactly once.")
        return self


class DimensionAssessment(StrictModel):
    dimension_id: str = Field(min_length=1)
    band: float = Field(ge=0, le=3, multiple_of=0.1)
    justification: str = Field(min_length=1)
    evidence_segment_ids: list[str] = Field(min_length=1)


class Assessment(StrictModel):
    component_assessments: list[DimensionAssessment] = Field(min_length=3, max_length=3)


class FeedbackPoint(StrictModel):
    message: str = Field(min_length=1)
    evidence_segment_ids: list[str] = Field(default_factory=list)


class FeedbackComponents(StrictModel):
    strengths: list[FeedbackPoint] = Field(default_factory=list)
    weaknesses: list[FeedbackPoint] = Field(default_factory=list)
    suggestions: list[FeedbackPoint] = Field(default_factory=list)


class GenerationOutput(StrictModel):
    assessment: Assessment
    feedback: FeedbackComponents