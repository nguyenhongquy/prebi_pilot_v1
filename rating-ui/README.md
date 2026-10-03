# PREBI Rating UI

Standalone browser-only annotation app for this repository. No research-workbench
or Python backend is required, and the annotation workflow makes no API or model
requests. Uploaded packet data stays in the browser; drafts and submitted ratings
are stored in localStorage until cleared. Treat the browser profile as protected
storage, and download results before clearing it.

## Run

From the repository root:

```bash
npm --prefix rating-ui install
npm --prefix rating-ui run dev
```

Open the localhost URL printed by Vite (normally http://127.0.0.1:5174).
Upload the blinded packet bundle, never the protected condition key. Enter an
assigned pseudonymous annotator code. The three task tabs separate assessment
quality, feedback quality, and feedback-implied score coding.

Feedback protocol `0.2.0` and assessment protocol `0.3.0` use three-point scales with criterion-specific anchors.
Feedback quality rates correctness and developmental usefulness; assessment
quality rates Score-Rubric Fit, Evidence Support, and Justification Quality.
Assessment `0.2.0` retains four criteria for legacy packets. Unable-to-judge is separate and requires a
reason. Legacy `0.1.0` packets retain their five-point scale. Feedback-implied
learner scores remain 0.0-3.0 in tenths, not quality ratings. Export a separate new
packet campaign to adopt a revised protocol; do not edit existing packets or mix
ratings across versions.

## Build

```bash
npm --prefix rating-ui run build
```

The static build is in `rating-ui/dist`. It contains code only. Never put research
packets, condition keys, or rating exports in this app directory or a public
static host. Distribute packets and collect downloaded ratings through an
approved protected channel. Browser drafts are origin-specific; changing host,
port, or browser profile does not transfer drafts. Existing drafts from the
research workbench origin are not automatically migrated.