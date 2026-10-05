from __future__ import annotations

import json
from typing import Any

from reflection_assessment_feedback.data import TARGET_COLUMNS
from reflection_assessment_feedback.models import ReflectionDocument, SegmentAnalysis
from reflection_assessment_feedback.segment_classification_gemini import SegmentClassifierGemini


HYBRID_CORRECTION_PROMPT = """## Hybrider Korrekturauftrag
Prüfe die nachstehende vorläufige automatische GBERT-Analyse anhand der vollständigen Reflexion und der Bewertungsrubrik.

Die GBERT-Vorhersage ist ein **Ausgangspunkt, kein Referenzlabel**. Korrigiere eine Vorhersage nur dann, wenn der Segmenttext im Kontext der vollständigen Reflexion und der Rubrik eine andere Zuordnung hinreichend begründet. Behalte plausible Vorhersagen bei. Führe keine Korrektur allein deshalb durch, weil eine andere Interpretation ebenfalls möglich wäre.

Prüfe besonders **Grenzfälle zwischen SW, UA und HA sowie benachbarten Leistungsstufen**.

### 1. Situationswahrnehmung (SW) vs. Ursachenanalyse (UA)
Entscheide nach der **Funktion der Aussage**, nicht nach einzelnen Signalwörtern.

- **SW:** Die Aussage wählt eine bedeutsame und/oder reflexionsrelevante Situation aus, beschreibt, rekonstruiert, beurteilt oder charakterisiert sie. Auch Erläuterungen und Bewertungen können SW sein, wenn sie primär dazu dienen, die beobachtete Situation nachvollziehbar darzustellen.
- **UA:** Die Aussage erklärt oder interpretiert, **warum oder wodurch** etwas geschieht bzw. geschehen ist, oder erschließt Ursachen, Zusammenhänge, Motive oder zugrunde liegende Mechanismen, die über die reine Beschreibung oder Charakterisierung hinausgehen.
**Wichtig:** Bewertende oder interpretativ klingende Formulierungen sind nicht automatisch UA. Eine Aussage wie „Die Diskussion verlief wenig aktiv“ charakterisiert zunächst die Situation und ist SW. Erst eine erklärende Beziehung wie „Die Diskussion verlief wenig aktiv, weil die Fragestellung kaum unterschiedliche Positionen zuließ“ begründet UA.

Prüffrage:

> **Beschreibt/charakterisiert die Aussage, was der Fall war → SW; erklärt sie, warum oder wodurch es so war → UA.**

### 2. Ursachenanalyse (UA) vs. Handlungsalternativen (HA)
Entscheide danach, **was erklärt oder begründet wird**.

- **UA:** Die Begründung erklärt eine beobachtete Situation, ihre Ursachen oder zugrunde liegenden Zusammenhänge.
- **HA:** Die Aussage formuliert eine zukünftige Handlung, Konsequenz oder Veränderung oder begründet, warum eine solche Handlung geeignet sein könnte.
Prüffrage:

> **Erklärt die Aussage die beobachtete Situation → UA; begründet oder formuliert sie zukünftiges Handeln → HA.**
Beispiele:

- **UA:** „Die geringe Beteiligung könnte darauf zurückzuführen sein, dass die Schüler ihre Rollen nicht kannten.“
- **HA:** „Beim nächsten Mal würde ich feste Rollen vergeben, damit die Schüler ihre Aufgaben kennen und sich gezielter beteiligen können.“
- **UA + HA:** „Da die Schüler ihre Rollen nicht kannten, würde ich beim nächsten Mal feste Rollen vergeben.“

### 3. Mehrere Dimensionen
SW, UA und HA sind **eigenständige Rubrikdimensionen und keine sich gegenseitig ausschließenden Klassen**.

Wenn ein Segment mehrere unterscheidbare Funktionen enthält, können mehrere Dimensionen gleichzeitig vorhanden sein.

Beispiel:

> „Die Aufgabe war nicht kognitiv aktivierend, da sie nur reproduktive Antworten verlangte.“
enthält eine Charakterisierung der Situation (SW) und eine Erklärung dafür (UA). Entsprechend können sowohl SW als auch UA einen Wert > 0 erhalten.

Ersetze daher nicht automatisch eine Dimension durch eine andere, wenn der Text beide Funktionen erfüllt.

### 4. Leistungsstufen prüfen
Erst nachdem entschieden wurde, **ob eine Dimension vorhanden ist**, bestimme ihre Leistungsstufe anhand der Rubrik.

Gehe dabei zweistufig vor:

1. **Dimension vorhanden?**
Wenn nein → `0`.
2. **Wenn vorhanden: Welche höchste durch den Text tatsächlich belegte Leistungsstufe wird erreicht?**
Vergib eine höhere Leistungsstufe nicht aufgrund von sprachlicher Komplexität oder Länge. Entscheidend sind ausschließlich die Anforderungen der jeweiligen Rubrikstufe.

Prüfe besonders benachbarte Stufen sorgfältig, insbesondere **UA 1 vs. UA 2**:

- **UA 1:** Eine Ursache oder ein Zusammenhang wird lediglich benannt oder behauptet.
- **UA 2:** Es liegt eine tatsächlich erklärende bzw. interpretierende Beziehung vor, auch wenn diese nicht zusätzlich durch Theorie, Core Reflections oder weitere Evidenz gestützt wird.

### 5. Kontext verwenden
Interpretiere jedes Segment im Kontext der vollständigen Reflexion. Nutze vorhergehende und nachfolgende Segmente, wenn sie erforderlich sind, um Referenzen, Begründungsbeziehungen oder die Funktion einer Aussage zu verstehen.

Bewerte jedoch nur das, was dem jeweiligen Segment zugerechnet werden kann. Übertrage nicht automatisch Analyse- oder Begründungsleistungen aus einem anderen Segment.

### 6. Korrekturprinzip
Für jedes GBERT-Label:

- **beibehalten**, wenn es durch Text, Kontext und Rubrik plausibel gestützt wird;
- **korrigieren**, wenn Text und Rubrik eine andere Zuordnung klarer stützen;
- bei einem echten Grenzfall **keine unnötige Korrektur erzwingen**.
Achte insbesondere auf folgende typische Fehler:

- SW wird fälschlich als UA erkannt, weil die Formulierung bewertend oder interpretativ klingt;
- UA wird fälschlich als SW erkannt, obwohl eine Ursache oder erklärende Beziehung formuliert wird;
- eine Aussage enthält tatsächlich SW **und** UA, GBERT erkennt aber nur eine Dimension;
- eine Begründung für eine zukünftige Handlung wird fälschlich als UA statt HA eingeordnet;
- UA 1 und UA 2 werden verwechselt, obwohl zwischen bloßer Behauptung und tatsächlicher Erklärung unterschieden werden kann.

### Ausgabe
Behalte die vorgegebenen Segmentgrenzen, die ursprüngliche Reihenfolge, die nullbasierten `index`-Werte sowie die JSON-Feldnamen und zulässigen Labelwerte unverändert.

Behalte explizite `scope_constraint`-Vorgaben bei.

Gib für **jedes Segment genau eine vollständige korrigierte Vorhersage** aus, einschließlich aller unveränderten Labels.

Gib ausschließlich das geforderte JSON aus.
"""


class SegmentClassifierHybrid(SegmentClassifierGemini):
    """Correct provisional GBERT labels using rubric-guided whole-reflection context."""

    def __init__(self, model: Any, baseline_predictor: Any, **options: Any) -> None:
        self.baseline_predictor = baseline_predictor
        super().__init__(model, **options)

    @classmethod
    def from_google_gemini(
        cls,
        *,
        model_name: str,
        temperature: float = 0.0,
        baseline_predictor: Any = None,
        **options: Any,
    ) -> SegmentClassifierHybrid:
        if baseline_predictor is None:
            raise ValueError("Hybrid classification requires a GBERT baseline predictor.")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as error:
            raise RuntimeError(
                "Install the optional Gemini dependencies with `uv sync --extra gemini`."
            ) from error
        model = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        return cls(model=model, baseline_predictor=baseline_predictor, **options)

    def _prompt(
        self,
        document: ReflectionDocument,
        *,
        gold_in_scope_segment_ids: set[str],
    ) -> str:
        texts = [segment.text for segment in document.segments]
        provisional = self.baseline_predictor.predict(texts).copy().reset_index(drop=True)
        if len(provisional) != len(document.segments):
            raise ValueError("GBERT analysis must cover every reflection segment.")
        forced_indices = [
            index for index, segment in enumerate(document.segments)
            if segment.segment_id in gold_in_scope_segment_ids
        ]
        if forced_indices:
            oracle = self.baseline_predictor.predict(
                [texts[index] for index in forced_indices], gold_in_scope=True,
            ).reset_index(drop=True)
            if len(oracle) != len(forced_indices):
                raise ValueError("GBERT oracle analysis must cover every constrained segment.")
            for position, index in enumerate(forced_indices):
                provisional.loc[index, ["scope", *TARGET_COLUMNS]] = oracle.loc[
                    position, ["scope", *TARGET_COLUMNS]
                ].to_numpy()
        automatic_analysis = []
        for index, segment in enumerate(document.segments):
            row = provisional.iloc[index]
            if row["scope"] not in (0, 1):
                raise ValueError("GBERT scope must be 0 or 1.")
            analysis = SegmentAnalysis.model_validate({
                "segment_id": segment.segment_id,
                "scope": "in_scope" if row["scope"] == 1 else "out_of_scope",
                "component_bands": {dimension: str(row[dimension]) for dimension in TARGET_COLUMNS},
            })
            automatic_analysis.append({
                "index": index,
                "scope": analysis.scope,
                "component_bands": analysis.component_bands,
            })
        return (
            super()._prompt(document, gold_in_scope_segment_ids=gold_in_scope_segment_ids)
            + "\n\n" + HYBRID_CORRECTION_PROMPT
            + "Vorlaeufige automatische GBERT-Analyse:\n"
            + json.dumps(automatic_analysis, ensure_ascii=False)
        )