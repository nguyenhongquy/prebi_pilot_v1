# Descriptive Statistics

Snapshot: **2026-10-06**. This report summarizes the saved author-disjoint modeling
splits, historical corpus annotation audit, matched LLM-judge evaluation, and
five-repeat generation consistency experiment. Results are descriptive pilot
evidence, not confirmatory findings. Only aggregate research statistics are shown.

## 1. Data Views and Author Identity

These views have different inclusion rules and must not be treated as one denominator.

| Data view | Documents | Authors | Segments / candidate rows |
| --- | ---: | ---: | --- |
| Full source-document registry | 242 | 64 canonical authors | Not a segment-level modeling dataset |
| Historical full annotation audit | 239 | 63 trim/case-folded author names | 11,625 segments |
| Current author-disjoint modeling splits | 239 | 62 canonical authors | 9,779 unique segments; 9,835 candidate rows |
| Generation / judge test slice | 18 | 5 canonical authors | 699 unique ordered segments |

The historical 63-author count used whitespace trimming and case folding. The
current pipeline also removes filename extensions and collapses whitespace,
merging an additional alias group in the annotated cohort. Raw filename aliases
are retained in restricted mappings but share canonical author IDs. The earlier
85 modeling author IDs were not normalized and are obsolete.

Source documents 108, 111, and 170 have no matching prepared annotation-candidate
rows and therefore never enter the modeling cohort. Their absence precedes the
text-length filter; the available artifacts do not establish why the annotation
input omitted them.

## 2. Author-Disjoint Modeling Splits

Splits use seed **20260929**. Authors, documents, and segments are disjoint across
train/dev/test. All candidates belonging to an underlying segment stay together.
Candidate rows preserve alternative annotation bundles; they are not independent
new text segments. Long modeling views contain three dimension rows per candidate.

| Split | Authors | Documents | Unique segments | Candidate rows | Feedback-available documents | Review-required candidate rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 48 | 184 | 7,081 | 7,124 | 0 | 86 |
| Dev | 8 | 31 | 1,795 | 1,806 | 0 | 22 |
| Test | 6 | 24 | 903 | 905 | 22 | 4 |
| **Total** | **62** | **239** | **9,779** | **9,835** | **22** | **112** |

### Eligibility and Exclusions

Modeling exports exclude text shorter than **10 characters after trimming**;
exactly 10 characters is allowed. Blank and punctuation-only text is also
excluded. The policy applies to all train/dev/test modeling-wide, modeling-long,
and scope views and is used when balancing train/dev groups. Excluded candidates
remain in separate audit artifacts. Unresolved scope and unknown in-scope skill
labels are excluded from the common modeling view.

| Split | Unresolved-scope rows excluded | Unknown-skill rows excluded | Text-quality rows excluded | Text-quality segments excluded | Total candidate rows excluded |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 78 | 1,321 | 289 | 217 | 1,688 |
| Dev | 10 | 190 | 37 | 32 | 237 |
| Test | 10 | 188 | 38 | 30 | 236 |
| **Total** | **98** | **1,699** | **364** | **279** | **2,161** |

Exclusion reasons can overlap; the total counts each candidate once. The prepared
input has **11,996 candidates across 11,805 underlying segments**; 9,835 retained
plus 2,161 excluded accounts for all candidates. Across the prepared input, 49
segments have competing scope candidates, 371 candidates require review, and
1,729 in-scope candidates initially lack skill targets. These pre-filter counts
are distinct from the mutually accounted modeling/exclusion summaries above.

### Scope Label Counts

| Split | Out of scope (0) | In scope (1) | Total |
| --- | ---: | ---: | ---: |
| Train | 343 | 6,781 | 7,124 |
| Dev | 83 | 1,723 | 1,806 |
| Test | 31 | 874 | 905 |
| **Total** | **457** | **9,378** | **9,835** |

### In-Scope Skill Label Counts

Each in-scope candidate has all three skill targets. `0` means the skill is
absent; `1` corresponds to original Stufe I/II, `2` to Stufe III, and `3` to
Stufe IV/V. Out-of-scope rows have non-applicable skill labels, not skill negatives.

| Split | Skill | 0 | 1 | 2 | 3 |
| --- | --- | ---: | ---: | ---: | ---: |
| Train | Situationserfassung (SW) | 4,029 | 2,256 | 475 | 21 |
| Train | Analyse (UA) | 3,819 | 1,989 | 636 | 337 |
| Train | Konsequenzen (HA) | 5,714 | 700 | 328 | 39 |
| Dev | SW | 1,017 | 565 | 136 | 5 |
| Dev | UA | 982 | 465 | 198 | 78 |
| Dev | HA | 1,447 | 197 | 70 | 9 |
| Test | SW | 480 | 314 | 80 | 0 |
| Test | UA | 533 | 216 | 97 | 28 |
| Test | HA | 735 | 81 | 50 | 8 |

Labels are strongly imbalanced. In particular, test SW has no class-3 examples;
a fixed-label macro-F1 that includes this class differs from a supported-label
macro-F1. Zero support is not evidence of performance on that class.

## 3. Historical Corpus and Annotation Descriptives

These statistics describe the historical **11,625-segment** audit, not the
current 9,779-segment modeling view or the 11,805-segment prepared candidate input.
The different exports should not be conflated into a single filtering flow.

| Statistic | Value |
| --- | ---: |
| Documents | 239 |
| Authors under historical trim/case-fold normalization | 63 |
| Segments | 11,625 |
| Seminars / semesters | 2 / 5 |
| Segments per document, median [IQR] | 47 [31-64] |
| Words per segment, median [IQR] | 18 [12-24] |
| Annotators, creator/editor union | 6 |
| Segments with one annotation creator | 10,694 (91.99%) |
| Segments with more than one annotation creator | 931 (8.01%) |
| Reflection-related segments | 11,164 (96.03%) |
| Non-reflection-related segments | 461 (3.97%) |

| Distinct annotation creators per segment | Segments | Share |
| --- | ---: | ---: |
| 1 | 10,694 | 91.99% |
| 2 | 927 | 7.97% |
| 3 | 4 | 0.03% |

Creator counts describe annotation provenance, not inter-rater agreement. The
six-annotator count also includes editors. Annotation effort is uneven: the two
largest pseudonymous creators account for 41.7% and 43.2% of single-creator
segments. No human agreement coefficient is inferred from these counts.

Applying only the 10-character policy to the historical export retains **11,420
segments** and excludes **205** short segments: 10,965 reflection-related and 455
non-reflection-related segments remain. This is a length-filter audit of that
historical export, not the current common modeling eligibility result.

## 4. Generation and Judge Evaluation Cohort

The classifier test set has 24 documents. The author of exploratory document
188 contributes four development/playground reflections: **3, 64, 126, 188**.
Exclude that complete author group, then explicitly exclude **218 and 221**, which
have no teacher-feedback records. This leaves **18 reference-complete test
reflections across five canonical authors**, with **699 ordered unique segments**.
Other authors' R1 reflections remain included; the rule is author-based, not
essay-based. Missing or ambiguous references in this selected slice cause an
error rather than silent exclusion.

- New intermediate analysis: author-disjoint **direct GBERT**, scope plus three
  direct skill heads, run `retrain-20261006T072742859011Z`.
- Generator: Google **`gemini-3.5-flash-lite`**, generation prompt **1.4.0** and
  G2 supplemental prompt **1.0.0**, rubric **1.0.0**. The model uses fixed sampling
  defaults and ignores the configured temperature.
- Judge: OpenAI **`gpt-6.1-sol`**; assessment-quality prompt **0.5.0** and
  feedback-quality prompt **0.3.0**.
- Judge inputs are condition-blinded and include the reflection, rubric, output,
  and teacher reference. Assessment judging also receives a candidate-aware
  provisional gold-analysis summary. Gold/reference information is not sent to
  the generation model.

The new G2 judge batch has **18 generated outputs, 36 judge records, and 90
criterion ratings**, with zero unable-to-judge ratings. Historical G1/G2/G3 each
have 36 test judge records; the four-way comparison covers **144 records and 360
criterion ratings**. Reflections, rubrics, teacher references, gold contexts,
judge identity, and prompt artifacts were checked before matching comparisons.
Historical outputs and new G2 remain separate; G1/G3 were not regenerated or
rejudged for this comparison. The partial earlier 16-document batch is not pooled.

### Mean Judge Ratings

Scale: **1-3**, larger is better. Every cell below uses 18 scorable document ratings.

| Task / criterion | Historical G1 | Historical G2 | New G2 | Historical G3 |
| --- | ---: | ---: | ---: | ---: |
| Assessment: score-rubric fit | 2.556 | 2.333 | 2.333 | 2.611 |
| Assessment: evidence support | 2.667 | 2.389 | 2.556 | 2.611 |
| Assessment: justification quality | 2.000 | 2.000 | 2.000 | 2.000 |
| Feedback: correctness | 2.444 | 2.333 | 2.667 | 2.722 |
| Feedback: developmental usefulness | 2.000 | 2.056 | 2.000 | 2.000 |

### Paired New-G2 Changes Against Historical G2

| Criterion | Mean difference | Improved / unchanged / worsened documents |
| --- | ---: | --- |
| Score-rubric fit | 0.000 | 2 / 14 / 2 |
| Evidence support | +0.167 | 6 / 9 / 3 |
| Justification quality | 0.000 | 0 / 18 / 0 |
| Correctness | +0.333 | 6 / 12 / 0 |
| Developmental usefulness | -0.056 | 0 / 17 / 1 |

These are ordinal-score descriptives, not significance tests. Regeneration and
judge variability prevent attribution of every change solely to author-disjoint
classifier training. Judge scores do not establish human-rated validity.

## 5. Five-Repeat Generation Consistency

This measures repeated **generation**, not repeated LLM judgments or feedback
text similarity. Five fresh generations per reflection yield SW/UA/HA assessment
scores. The new experiment produced **90 fresh G2 outputs** and reused **270
validated historical test outputs**, totaling **360 outputs and 1,080 dimension
scores**. Every condition/dimension has all five repeats for all 18 reflections,
with no missing ratings. Historical playground outputs are not pooled into test.
The judge batch's single output is not counted as a consistency repetition.

Scores are exact decimal categories from 0.0 to 3.0 in 0.1 increments, without
rounding. Pairwise exact agreement averages the ten within-reflection repetition
pairs. Unanimity is the fraction of reflections with all five scores identical.
Krippendorff alpha uses repetition rounds as raters and reflections as units;
nominal alpha is primary for the exact-category policy, ordinal alpha supplementary.

### Pairwise Exact Agreement

| Dimension | Historical G1 | Historical G2 | New G2 | Historical G3 |
| --- | ---: | ---: | ---: | ---: |
| SW | 0.7944 | 0.6111 | 0.5556 | 0.8056 |
| UA | 0.5333 | 0.3556 | 0.4222 | 0.3500 |
| HA | 0.5056 | 0.3667 | 0.3722 | 0.4278 |

### Unanimous Agreement Rate

| Dimension | Historical G1 | Historical G2 | New G2 | Historical G3 |
| --- | ---: | ---: | ---: | ---: |
| SW | 0.6667 | 0.2778 | 0.3333 | 0.6111 |
| UA | 0.1111 | 0.0556 | 0.1667 | 0.0000 |
| HA | 0.2222 | 0.0556 | 0.0556 | 0.1111 |

### Nominal Krippendorff Alpha

| Dimension | Historical G1 | Historical G2 | New G2 | Historical G3 |
| --- | ---: | ---: | ---: | ---: |
| SW | 0.7058 | 0.5384 | 0.4645 | 0.7110 |
| UA | 0.3938 | 0.2670 | 0.3432 | 0.1997 |
| HA | 0.3643 | 0.2887 | 0.3102 | 0.3267 |

### Ordinal Krippendorff Alpha

| Dimension | Historical G1 | Historical G2 | New G2 | Historical G3 |
| --- | ---: | ---: | ---: | ---: |
| SW | 0.9461 | 0.9086 | 0.8503 | 0.8866 |
| UA | 0.8695 | 0.8227 | 0.8883 | 0.7078 |
| HA | 0.8519 | 0.7767 | 0.8616 | 0.8849 |

### Paired Repeatability Differences: New G2 Minus Historical G2

| Dimension | Exact-agreement difference | Exploratory 95% bootstrap interval |
| --- | ---: | --- |
| SW | -0.0556 | [-0.2316, 0.1333] |
| UA | +0.0667 | [-0.0737, 0.2579] |
| HA | +0.0056 | [-0.1889, 0.2118] |

Intervals use **five corrected author clusters**, 2,000 bootstrap samples, and
seed 20260929. All three intervals include zero. With few clusters and historical
generation dates, these are exploratory rather than confirmatory intervals.
New G2 does not consistently improve repeatability: SW exact agreement declines,
UA improves, and HA is nearly unchanged. High ordinal alpha can coexist with
lower exact agreement when score changes are small. Stability is not correctness.

## 6. Saved Sources and Validation

- Current split counts: [provisional_split_summary.csv](data/provisional_segment_splits/provisional_split_summary.csv).
- Current label counts: [provisional_split_label_summary.csv](data/provisional_segment_splits/provisional_split_label_summary.csv).
- Author-disjoint export provenance: [author_disjoint_provenance.json](data/provisional_segment_splits/author_disjoint_provenance.json).
- Historical annotation audit: [corpus_annotation_summary.json](artifacts/corpus_annotation_descriptives/corpus_annotation_summary.json). Its pre-fix `current_split_author_ids` field is stale; use the current split manifest instead.
- Judge comparison means: [historical-comparison-summary.csv](artifacts/g2-author-disjoint-judge/c9c3a10dfdd8756be0206e93e5025af7d3cd2629ed3a34397db88f6ebfdabf6a/historical-comparison-summary.csv).
- Judge paired differences: [historical-comparison-paired-differences.csv](artifacts/g2-author-disjoint-judge/c9c3a10dfdd8756be0206e93e5025af7d3cd2629ed3a34397db88f6ebfdabf6a/historical-comparison-paired-differences.csv).
- Consistency metrics: [within.csv](artifacts/g2-generation-consistency/f73bbd43a95d1b6692049cb88fc1cb106dcddf01c85fea26663016926aa92ca4/within.csv).
- Consistency intervals: [paired-differences.csv](artifacts/g2-generation-consistency/f73bbd43a95d1b6692049cb88fc1cb106dcddf01c85fea26663016926aa92ca4/paired-differences.csv).
- Analysis notebooks: [09_llm_judge_analysis.ipynb](09_llm_judge_analysis.ipynb) and [10_generation_consistency.ipynb](10_generation_consistency.ipynb).

Saved artifact provenance, cohort membership, complete repetition coverage,
evidence identifiers, judge schemas, and matched input snapshots were validated.
The latest workflow validation passed **101 tests** with external tracing disabled
for unit tests. Full text, generated feedback, teacher references, and judge
rationales remain in protected storage and are not reproduced here.