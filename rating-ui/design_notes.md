
# PReBi UI Redesign Brief

## Context

PReBi is a research prototype for AI-supported feedback on student
reflective writing at Pädagogische Hochschule Ludwigsburg (PHL).

The application currently supports expert evaluation of generated
assessment and feedback. Later, the same UI architecture should support
lecturers reviewing and editing generated feedback before it is given
to students.

This is a first-year evolutionary prototype maintained by one developer.
Prefer simple, reusable components over architectural complexity.

IMPORTANT:
The existing packet format, local browser persistence, blinding,
randomization, rating data, import/export behavior, and evaluation logic
must continue to work. Redesign the UI without unnecessarily changing
the underlying research logic.

# 1. Application architecture

Use ONE shared PReBi application shell with task-specific workspaces.

Current workflow:

- Expert evaluation

Future workflow:

- Lecturer feedback review

Do not combine expert rating controls and lecturer editing controls
on the same screen.

Design reusable components so that both workflows can later share:

- AppShell
- PacketNavigator
- ReflectionViewer
- EvidenceHighlighter
- OutputViewer
- TaskPanel

Expert-specific components can include:

- RatingCriterion
- RatingScale
- RatingProgress

Future lecturer-specific components may include:

- FeedbackEditor
- ReviewDecision

# 2. Corporate design

Follow the new PH Ludwigsburg corporate design.

Typography:

- Figtree for UI and body text
- Figtree Fett / bold for headings and emphasis
- Arial as fallback
- Avoid serif fonts
- Body text should be black
- Left-align normal text

Colors:

--ph-moss: #002d20;
--ph-green: #115933;
--ph-leaf: #b1edbf;
--ph-offwhite: #e7ede6;
--ph-white: #ffffff;
--ph-black: #000000;
--ph-lilac: #d7d9ff;
--ph-purple: #4309bb;

Usage:

- Off-white = application canvas/background
- White = main reading and task surfaces/cards
- Moss/PH green = identity, headings, selected states, primary actions
- Leaf green = evidence highlighting / subtle positive highlighting
- Lilac/purple = optional secondary/focus accent only
- Do not create an overall pale-green tint
- Do not overuse accent colors

Accessibility:

- Target WCAG AA
- Visible keyboard focus
- Adequate control sizes
- Do not communicate state through color alone
- Maintain readable line lengths, ideally about 35–80 characters for
  continuous prose

# 3. Global application shell

Create a restrained institutional application header.

Header:
[PH Ludwigsburg logo]
PReBi
Feedback for reflective writing

Secondary information:
Expert evaluation
Current task
Progress

Do NOT make "PReBi" itself look like a logo.

Reduce permanent utility controls in the header.
Upload/export/clear-data are secondary file-management actions rather
than the primary annotation task.

# 4. Expert evaluation start screen

Before a packet is loaded, show a simple onboarding screen rather than
the empty annotation workspace.

Content:

PReBi
Expert evaluation

"Evaluate automatically generated assessment and feedback for student
reflections. Generator information is blinded."

Annotator code
[input]

[Open evaluation packet]

"Ratings and drafts are stored locally in this browser."

The primary action should be obvious.
Do not show disabled annotation controls before a packet is loaded.

# 5. Expert evaluation workspace

The central design goal is:

KEEP THE SOURCE MATERIAL VISIBLE WHILE THE EXPERT IS RATING IT.

Avoid one very long page requiring repeated scrolling between:
reflection -> generated output -> rating criterion.

Desktop layout:

---

Header / task / progress
------------------------

Packet nav | Reflection | Generated output
           |            |
           |            |
-------------------------

Current rating criterion
------------------------

The upper comparison area should remain available while rating.

Suggested proportions:

- narrow packet/navigation column
- substantial reflection column
- substantial generated-output column

Reflection and output areas may scroll independently if necessary.

# 6. Packet navigation

Show:

- Packet X of N
- previous / next
- completion state
- overall progress

Example:

Packet 3 of 18

✓ 1
✓ 2
● 3
○ 4
○ 5

Rating guide >
Rubric >

Do not expose model/generator condition information.
Preserve experimental blinding.

# 7. Reflection viewer

Optimize the reflection for reading rather than filling available width.

Requirements:

- readable line length
- generous line spacing
- segment numbers remain visible
- segment numbers should be visually secondary
- allow evidence segments to be highlighted

Example:

33   Die Schüler*innen konnten im Austausch
     ihre eigenen Vorstellungen diskutieren ...

34   Zudem konnte ein Unterschied ...

35   Es war sowohl mehr Fachwissen ...

# 8. Evidence interaction

Generated assessment/feedback already references reflection segments.

Replace technical identifiers such as long segment UUIDs in the visible
UI with human-readable segment numbers where possible.

Example:

Evidence: [33] [35]

Clicking or hovering an evidence reference should visually identify the
corresponding reflection segment.

Use PH leaf green (#b1edbf) as a restrained evidence highlight where
accessible.

Do not alter the stored underlying segment IDs.

# 9. Generated assessment output

Present generated assessment dimensions as readable cards/sections.

Example:

SW                                      Score 2.0

Die Reflexion beschreibt und rekonstruiert ...

Evidence: [78] [80]

UA                                      Score 2.0

Die Verfasserin formuliert ...

Evidence: [81]

Prioritize readable text over fitting many cards horizontally.

# 10. Generated feedback output

Do NOT use three very narrow text columns for:

Strengths | Development needs | Next-step suggestions

The current columns make long feedback difficult to read.

Prefer a readable vertical presentation:

STRENGTHS
[feedback]
Evidence: [33] [35]

DEVELOPMENT NEEDS
[feedback]
Evidence: [36] [37]

NEXT-STEP SUGGESTIONS
[feedback]

# 11. Rating interaction

Reduce the feeling of one enormous form.

Prefer presenting ONE rating criterion prominently at a time while
keeping source material accessible.

Example:

YOUR EVALUATION

1 Score–rubric fit       ●
2 Evidence support       ○
3 Justification quality  ○

---

Evidence support

Check whether cited evidence supports the central judgments.

[ 1 ]   [ 2 ]   [ 3 ]             [ ] Unable to judge

1  Central judgments rely on absent...
2  Some evidence supports...
3  Cited evidence exists and supports...

Comment (optional)
[                                      ]

[Previous criterion]            [Save & next]

Preserve all existing rating dimensions and anchors exactly.
Do not rewrite the research rubric unless explicitly requested.

# 12. Completion screen

After all required ratings are completed, show a clear finish state.

Evaluation complete

✓ 18 / 18 packets
✓ All required ratings complete

[Export ratings]

Explain that responses remain locally stored until exported.

Make export prominent here rather than permanently competing with the
annotation task.

# 13. Future lecturer-review mode

Do not implement this unless requested, but design components so it can
be added later.

Expected layout:

Reflection | Draft feedback
           | editable feedback

Review decision:

- Use as generated
- Use with edits
- Do not use

[Approve & next]

This mode should share the AppShell, ReflectionViewer,
EvidenceHighlighter and PacketNavigator with expert evaluation but
should NOT contain research rating controls.

# 14. Implementation constraints

Before changing code:

1. Inspect the current project structure and existing components.
2. Identify which components and state contain research/application
   logic versus presentation logic.
3. Propose the smallest refactoring necessary for this redesign.
4. Do not change packet schemas or stored evaluation data unless
   explicitly requested.
5. Do not change blinding/randomization behavior.
6. Do not replace working local persistence.
7. Do not introduce a backend.
8. Do not introduce a large UI framework unless already used.
9. Prefer semantic HTML and simple reusable components.
10. Preserve existing functionality while progressively replacing the UI.

First report:

- relevant existing components
- proposed component structure
- files that need modification
- functionality at risk of regression

Do not start a large rewrite before reporting this plan.
