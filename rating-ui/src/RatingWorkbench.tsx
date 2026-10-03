import { useEffect, useMemo, useState, type ChangeEvent } from 'react'
import {
  Check,
  ChevronLeft,
  ChevronRight,
  Download,
  FileJson,
  Save,
  ShieldCheck,
  Trash2,
  Upload,
} from 'lucide-react'
import './RatingWorkbench.css'

type RatingTask = 'assessment_quality' | 'feedback_quality' | 'feedback_implied_score'
type Criterion = { criterion_id: string; label: string; description: string; anchors?: Record<string, string> }
type Segment = { segment_id: string; order: number; text: string }
type FeedbackItem = { text: string; evidence_segment_ids?: string[] }
type EvidenceSpan = { start_character: number; end_character: number; quote: string }
type Packet = {
  packet_id: string
  task: RatingTask
  packet_version: string
  protocol_id: string
  protocol_version: string
  human_protocol: string
  criteria?: Criterion[]
  scale?: { min: number; max: number; anchors: Record<string, string> }
  score_dimensions?: Array<{ dimension_id: string; label: string }>
  reflection: { segments: Segment[] }
  rubric: { version: string; dimensions: Array<{ dimension_id: string; name: string; description: string; bands: Record<string, string> }> }
  output?: {
    dimensions?: Array<{ dimension_id: string; score: number; justification: string; evidence_segment_ids: string[] }>
    strengths?: FeedbackItem[]
    weaknesses?: FeedbackItem[]
    suggestions?: FeedbackItem[]
  }
  human_feedback?: string
  display_output_sha256?: string
  displayed_reflection_sha256?: string
  condition_blinded: true
}
type CriterionAnswer = { score: number | null; unable_to_judge: boolean; comment: string }
type SpanComment = {
  feedback_component: string
  feedback_item_index: number
  start_character: number
  end_character: number
  comment: string
  tags: string[]
  linked_reflection_segment_ids: string[]
}
type ScoreAnswer = {
  status: 'inferred_from_feedback' | 'not_inferable'
  score: number | null
  confidence: 'low' | 'medium' | 'high' | 'not_inferable'
  rationale: string
  feedback_evidence: EvidenceSpan[]
}
type Draft = {
  criterionRatings: Record<string, CriterionAnswer>
  scoreAnswers: Record<string, ScoreAnswer>
  overallComment: string
  spanComments: SpanComment[]
}
type SavedRating = {
  schema_version: '1.0.0'
  rating_id: string
  packet_id: string
  task: RatingTask
  evaluator_type: 'human'
  evaluator_id: string
  protocol_id: string
  protocol_version: string
  packet_version: string
  display_output_sha256?: string
  displayed_reflection_sha256?: string
  submitted_at: string
  response: unknown
}

const TASK_LABELS: Record<RatingTask, string> = {
  feedback_implied_score: 'Feedback-implied score',
  assessment_quality: 'Assessment quality',
  feedback_quality: 'Feedback quality',
}
const STORAGE_KEY = 'prebi-rating-workbench-v1'
const TAGS = ['unsupported', 'inaccurate', 'vague', 'actionable', 'strength', 'concern', 'other']

function emptyDraft(packet: Packet): Draft {
  return {
    criterionRatings: Object.fromEntries((packet.criteria ?? []).map((item) => [item.criterion_id, { score: null, unable_to_judge: false, comment: '' }])),
    scoreAnswers: Object.fromEntries((packet.score_dimensions ?? []).map((item) => [item.dimension_id, { status: 'not_inferable', score: null, confidence: 'not_inferable', rationale: '', feedback_evidence: [] }])),
    overallComment: '',
    spanComments: [],
  }
}

function queueKey(packet: Packet, evaluatorId: string) {
  return `${evaluatorId.trim()}::${packet.packet_id}`
}

function downloadJson(filename: string, value: unknown) {
  const blob = new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

function isPacket(value: unknown): value is Packet {
  if (!value || typeof value !== 'object') return false
  const packet = value as Partial<Packet>
  const commonFields = typeof packet.packet_id === 'string'
    && ['assessment_quality', 'feedback_quality', 'feedback_implied_score'].includes(String(packet.task))
    && packet.condition_blinded === true
    && typeof packet.human_protocol === 'string'
    && Array.isArray(packet.reflection?.segments)
    && Array.isArray(packet.rubric?.dimensions)
    && typeof packet.rubric.version === 'string'
    && typeof packet.protocol_id === 'string'
    && typeof packet.protocol_version === 'string'
  if (!commonFields) return false
  if (packet.task === 'assessment_quality') return Array.isArray(packet.criteria) && Array.isArray(packet.output?.dimensions)
  if (packet.task === 'feedback_quality') return Array.isArray(packet.criteria) && ['strengths', 'weaknesses', 'suggestions'].every((key) => Array.isArray(packet.output?.[key as keyof NonNullable<Packet['output']>]))
  return typeof packet.human_feedback === 'string' && Array.isArray(packet.score_dimensions)
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`
  if (value && typeof value === 'object') {
    const object = value as Record<string, unknown>
    return `{${Object.keys(object).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(object[key])}`).join(',')}}`
  }
  return JSON.stringify(value)
}

async function sha256(value: unknown) {
  const bytes = new TextEncoder().encode(canonicalJson(value))
  const digest = await crypto.subtle.digest('SHA-256', bytes)
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('')
}

export default function RatingWorkbench() {
  const [packets, setPackets] = useState<Packet[]>([])
  const [packetIndex, setPacketIndex] = useState(0)
  const [taskFilter, setTaskFilter] = useState<RatingTask>('feedback_implied_score')
  const [evaluatorId, setEvaluatorId] = useState(() => localStorage.getItem('prebi-rating-evaluator') ?? '')
  const [drafts, setDrafts] = useState<Record<string, Draft>>(() => {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}') as Record<string, Draft> }
    catch { return {} }
  })
  const [submitted, setSubmitted] = useState<Record<string, SavedRating>>(() => {
    try { return JSON.parse(localStorage.getItem(`${STORAGE_KEY}-submitted`) ?? '{}') as Record<string, SavedRating> }
    catch { return {} }
  })
  const [error, setError] = useState('')
  const [status, setStatus] = useState('Upload the protected packet bundle to begin.')
  const [selectedSpan, setSelectedSpan] = useState<{ component: string; index: number; start: number; end: number; quote: string } | null>(null)
  const [spanComment, setSpanComment] = useState('')
  const [spanTags, setSpanTags] = useState<string[]>([])

  const taskPackets = useMemo(() => packets.filter((packet) => packet.task === taskFilter), [packets, taskFilter])
  const packet = taskPackets[packetIndex] ?? null
  const currentKey = packet && evaluatorId.trim() ? queueKey(packet, evaluatorId) : ''
  const draft = packet ? (drafts[currentKey] ?? emptyDraft(packet)) : null
  const doneCount = taskPackets.filter((item) => Boolean(submitted[queueKey(item, evaluatorId)])).length

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(drafts))
  }, [drafts])

  useEffect(() => {
    localStorage.setItem(`${STORAGE_KEY}-submitted`, JSON.stringify(submitted))
  }, [submitted])

  useEffect(() => {
    localStorage.setItem('prebi-rating-evaluator', evaluatorId)
  }, [evaluatorId])

  useEffect(() => {
    setPacketIndex(0)
    setSelectedSpan(null)
  }, [taskFilter])

  function updateDraft(patch: Partial<Draft>) {
    if (!packet || !currentKey || !draft) return
    setDrafts((current) => ({ ...current, [currentKey]: { ...draft, ...patch } }))
  }

  async function importPacketFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    setError('')
    try {
      const parsed: unknown = JSON.parse(await file.text())
      const loaded = Array.isArray(parsed)
        ? parsed
        : parsed && typeof parsed === 'object' && Array.isArray((parsed as { packets?: unknown }).packets)
          ? (parsed as { packets: unknown[] }).packets
          : [parsed]
      if (!loaded.every(isPacket)) throw new Error('File must contain blinded rating packets.')
      const seen = new Set<string>()
      for (const item of loaded as Packet[]) {
        if (seen.has(item.packet_id)) throw new Error('Packet IDs must be unique in the uploaded bundle.')
        seen.add(item.packet_id)
        if (item.displayed_reflection_sha256 && await sha256(item.reflection) !== item.displayed_reflection_sha256) {
          throw new Error('A packet reflection hash does not match its displayed content.')
        }
        if (item.display_output_sha256 && await sha256(item.output ?? item.human_feedback) !== item.display_output_sha256) {
          throw new Error('A packet output hash does not match its displayed content.')
        }
      }
      const shuffled = [...loaded as Packet[]].sort(() => Math.random() - 0.5)
      setPackets(shuffled)
      setPacketIndex(0)
      const firstTask = (shuffled[0]?.task ?? 'feedback_implied_score') as RatingTask
      setTaskFilter(firstTask)
      setStatus(`Loaded ${shuffled.length} blinded packets. Order randomized for this session.`)
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Could not read packet file.')
    }
  }

  function submitRating() {
    if (!packet || !draft || !evaluatorId.trim()) {
      setError('Enter your annotator code before submitting.')
      return
    }
    if (packet.task === 'feedback_implied_score') {
      for (const criterion of packet.score_dimensions ?? []) {
        const answer = draft.scoreAnswers[criterion.dimension_id]
        if (!answer || !answer.rationale.trim()) {
          setError(`Add a rationale for ${criterion.label}.`)
          return
        }
        if (answer.status === 'inferred_from_feedback' && (answer.score === null || answer.confidence === 'not_inferable')) {
          setError(`Complete the score and confidence for ${criterion.label}.`)
          return
        }
        if (answer.status === 'not_inferable' && answer.feedback_evidence.length) {
          setError(`Remove score evidence for not-inferable ${criterion.label}.`)
          return
        }
      }
    } else {
      for (const criterion of packet.criteria ?? []) {
        const answer = draft.criterionRatings[criterion.criterion_id]
        if (!answer || (!answer.unable_to_judge && answer.score === null)) {
          setError(`Rate ${criterion.label}, or mark it unable to judge.`)
          return
        }
        if (answer.unable_to_judge && !answer.comment.trim()) {
          setError(`Explain why ${criterion.label} cannot be judged.`)
          return
        }
        if (!answer.unable_to_judge && (!Number.isInteger(answer.score) || answer.score! < (packet.scale?.min ?? 1) || answer.score! > (packet.scale?.max ?? 5))) {
          setError(`Choose a score within this packet's scale for ${criterion.label}.`)
          return
        }
      }
    }
    const record: SavedRating = {
      schema_version: '1.0.0',
      rating_id: crypto.randomUUID(),
      packet_id: packet.packet_id,
      task: packet.task,
      evaluator_type: 'human',
      evaluator_id: evaluatorId.trim(),
      protocol_id: packet.protocol_id,
      protocol_version: packet.protocol_version,
      packet_version: packet.packet_version,
      display_output_sha256: packet.display_output_sha256,
      displayed_reflection_sha256: packet.displayed_reflection_sha256,
      submitted_at: new Date().toISOString(),
      response: packet.task === 'feedback_implied_score'
        ? {
            scores: Object.entries(draft.scoreAnswers).map(([dimension_id, answer]) => ({ dimension_id, ...answer })),
            feedback_span_comments: draft.spanComments,
          }
        : { criterion_ratings: draft.criterionRatings, overall_comment: draft.overallComment, feedback_span_comments: draft.spanComments },
    }
    setSubmitted((current) => ({ ...current, [currentKey]: record }))
    setError('')
    setStatus('Rating submitted and saved in this browser. Download the result JSON for protected collection.')
  }

  function addSpanComment() {
    if (!selectedSpan || !spanComment.trim()) return
    const comment: SpanComment = {
      feedback_component: selectedSpan.component,
      feedback_item_index: selectedSpan.index,
      start_character: selectedSpan.start,
      end_character: selectedSpan.end,
      comment: spanComment.trim(),
      tags: spanTags,
      linked_reflection_segment_ids: [],
    }
    updateDraft({ spanComments: [...(draft?.spanComments ?? []), comment] })
    setSpanComment('')
    setSpanTags([])
    setSelectedSpan(null)
  }

  function handleFeedbackSelection(event: React.SyntheticEvent<HTMLTextAreaElement>, component: string, index: number) {
    const target = event.currentTarget
    const start = target.selectionStart
    const end = target.selectionEnd
    if (start < end) setSelectedSpan({ component, index, start, end, quote: target.value.slice(start, end) })
  }

  function downloadCurrent() {
    if (!packet || !currentKey) return
    const rating = submitted[currentKey]
    if (!rating) return
    downloadJson(`rating-${rating.rating_id}.json`, rating)
  }

  function downloadCompleted() {
    const records = taskPackets.map((item) => submitted[queueKey(item, evaluatorId)]).filter(Boolean)
    if (!records.length) return
    downloadJson(`ratings-${taskFilter}.json`, { schema_version: '1.0.0', ratings: records })
  }

  function clearLocalData() {
    if (!window.confirm('Clear all local drafts and submitted ratings from this browser? Download completed ratings first.')) return
    setDrafts({})
    setSubmitted({})
    setStatus('Local drafts and submissions cleared.')
  }

  return <div className="rating-shell">
    <header className="rating-header">
      <div><span className="rating-eyebrow">Blinded annotation</span><h1>PREBI Rating</h1></div>
      <label className="annotator-field">Annotator code<input value={evaluatorId} onChange={(event) => setEvaluatorId(event.target.value)} placeholder="Your assigned code" autoComplete="off" /></label>
    </header>
    <div className="rating-toolbar">
      <div className="task-tabs" role="tablist" aria-label="Rating task">
        {(Object.keys(TASK_LABELS) as RatingTask[]).map((task) => <button key={task} role="tab" aria-selected={taskFilter === task} className={taskFilter === task ? 'active' : ''} onClick={() => setTaskFilter(task)}>{TASK_LABELS[task]}</button>)}
      </div>
      <div className="rating-actions">
        <label className="icon-button" title="Upload packet JSON"><Upload size={16} /><span>Upload packets</span><input type="file" accept="application/json,.json" onChange={importPacketFile} /></label>
        <button className="icon-button" onClick={downloadCompleted} disabled={!doneCount} title="Download completed ratings"><Download size={16} /><span>Export completed ({doneCount})</span></button>
        <button className="icon-button danger" onClick={clearLocalData} title="Clear local drafts and submitted ratings"><Trash2 size={16} /><span>Clear local data</span></button>
      </div>
    </div>
    {error && <div className="rating-error" role="alert">{error}</div>}
    <div className="rating-status"><ShieldCheck size={16} />{status} Drafts and submissions stay in this browser until downloaded.</div>
    {!packet ? <div className="rating-empty"><FileJson size={34} /><strong>No packet for this task</strong><span>Upload the protected packet bundle to start annotating.</span></div> : <div className="rating-layout">
      <aside className="packet-rail">
        <div className="rail-heading"><strong>{TASK_LABELS[taskFilter]}</strong><span>{packetIndex + 1} / {taskPackets.length}</span></div>
        <div className="packet-progress"><span style={{ width: `${taskPackets.length ? doneCount / taskPackets.length * 100 : 0}%` }} /></div>
        <p className="blind-note"><ShieldCheck size={14} />Condition and model are hidden.</p>
        <div className="packet-nav">
          <button className="icon-button" disabled={packetIndex === 0} onClick={() => setPacketIndex((index) => Math.max(0, index - 1))} aria-label="Previous packet"><ChevronLeft size={18} /></button>
          <span>Packet {packetIndex + 1}</span>
          <button className="icon-button" disabled={packetIndex >= taskPackets.length - 1} onClick={() => setPacketIndex((index) => Math.min(taskPackets.length - 1, index + 1))} aria-label="Next packet"><ChevronRight size={18} /></button>
        </div>
        <details className="protocol-details"><summary>Rating guide · {packet.protocol_version}</summary><pre>{packet.human_protocol}</pre></details>
      </aside>
      <main className="rating-main">
        <section className="context-panel">
          <div className="panel-title"><span>Reflection</span><small>{packet.reflection.segments.length} segments</small></div>
          <div className="segments">{packet.reflection.segments.map((segment) => <article className="segment" key={segment.segment_id}><span className="segment-id">{segment.order + 1}</span><p>{segment.text}</p></article>)}</div>
          <details className="rubric-details"><summary>Rubric · {packet.rubric.version}</summary>{packet.rubric.dimensions.map((dimension) => <div className="rubric-dimension" key={dimension.dimension_id}><strong>{dimension.dimension_id} · {dimension.name}</strong><p>{dimension.description}</p><dl>{Object.entries(dimension.bands).map(([band, description]) => <div key={band}><dt>{band}</dt><dd>{description}</dd></div>)}</dl></div>)}</details>
        </section>
        {packet.task === 'feedback_implied_score' ? <ScoreCoding packet={packet} draft={draft!} updateDraft={updateDraft} onSelection={handleFeedbackSelection} selectedSpan={selectedSpan} spanComment={spanComment} setSpanComment={setSpanComment} addSpanComment={addSpanComment} /> : <QualityRating packet={packet} draft={draft!} updateDraft={updateDraft} onSelection={handleFeedbackSelection} selectedSpan={selectedSpan} spanComment={spanComment} setSpanComment={setSpanComment} spanTags={spanTags} setSpanTags={setSpanTags} addSpanComment={addSpanComment} />}
        <footer className="rating-footer">
          <span><Save size={15} />Autosaved locally</span>
          <div>{submitted[currentKey] && <button className="icon-button" onClick={downloadCurrent}><Download size={16} />Download this rating</button>}<button className="submit-button" onClick={submitRating}><Check size={16} />{submitted[currentKey] ? 'Update rating' : 'Submit rating'}</button></div>
        </footer>
      </main>
    </div>}
  </div>
}

function ScoreCoding({ packet, draft, updateDraft, onSelection, selectedSpan, spanComment, setSpanComment, addSpanComment }: {
  packet: Packet; draft: Draft; updateDraft: (patch: Partial<Draft>) => void
  onSelection: (event: React.SyntheticEvent<HTMLTextAreaElement>, component: string, index: number) => void
  selectedSpan: { component: string; index: number; start: number; end: number; quote: string } | null
  spanComment: string; setSpanComment: (value: string) => void; addSpanComment: () => void
}) {
  return <section className="task-panel">
    <div className="panel-title"><span>Human feedback</span><small>Code what the feedback implies; do not infer missing scores</small></div>
    <textarea className="feedback-source" readOnly value={packet.human_feedback ?? ''} onSelect={(event) => onSelection(event, 'human_feedback', 0)} rows={5} />
    {selectedSpan?.component === 'human_feedback' && <div className="span-editor"><strong>Selected evidence: “{selectedSpan.quote}”</strong><textarea value={spanComment} onChange={(event) => setSpanComment(event.target.value)} placeholder="Evidence note for this excerpt" rows={2} /><button className="icon-button" onClick={addSpanComment}>Add evidence note</button></div>}
    <div className="score-grid">{(packet.score_dimensions ?? []).map((dimension) => {
      const answer = draft.scoreAnswers[dimension.dimension_id]
      return <article className="score-card" key={dimension.dimension_id}>
        <header><span>{dimension.dimension_id}</span><strong>{dimension.label}</strong></header>
        <label className="infer-toggle"><input type="checkbox" checked={answer.status === 'inferred_from_feedback'} onChange={(event) => updateDraft({ scoreAnswers: { ...draft.scoreAnswers, [dimension.dimension_id]: event.target.checked ? { ...answer, status: 'inferred_from_feedback', confidence: 'medium' } : { status: 'not_inferable', score: null, confidence: 'not_inferable', rationale: answer.rationale, feedback_evidence: [] } } })} />Score inferable from feedback</label>
        {answer.status === 'inferred_from_feedback' && <div className="two-column"><label className="field"><span>Implied score <small>0.0–3.0 in tenths</small></span><input type="number" min="0" max="3" step="0.1" value={answer.score ?? ''} onChange={(event) => updateDraft({ scoreAnswers: { ...draft.scoreAnswers, [dimension.dimension_id]: { ...answer, score: event.target.value === '' ? null : Number(event.target.value) } } })} /></label><label className="field"><span>Confidence</span><select value={answer.confidence} onChange={(event) => updateDraft({ scoreAnswers: { ...draft.scoreAnswers, [dimension.dimension_id]: { ...answer, confidence: event.target.value as ScoreAnswer['confidence'] } } })}><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option></select></label></div>}
        <label className="field"><span>Rationale</span><textarea rows={3} value={answer.rationale} onChange={(event) => updateDraft({ scoreAnswers: { ...draft.scoreAnswers, [dimension.dimension_id]: { ...answer, rationale: event.target.value } } })} /></label>
        {selectedSpan?.component === 'human_feedback' && <button className="text-button use-evidence" onClick={() => { updateDraft({ scoreAnswers: { ...draft.scoreAnswers, [dimension.dimension_id]: { ...answer, feedback_evidence: [...answer.feedback_evidence, { start_character: selectedSpan.start, end_character: selectedSpan.end, quote: selectedSpan.quote }] } } }); }}>Use selected excerpt as evidence</button>}
        {answer.feedback_evidence.map((evidence, index) => <p className="evidence-chip" key={`${evidence.start_character}-${index}`}>“{evidence.quote}” <button className="text-button" onClick={() => updateDraft({ scoreAnswers: { ...draft.scoreAnswers, [dimension.dimension_id]: { ...answer, feedback_evidence: answer.feedback_evidence.filter((_, itemIndex) => itemIndex !== index) } } })}>Remove</button></p>)}
      </article>
    })}</div>
    <p className="annotation-note">Selecting text in the feedback marks an evidence span. If no defensible score is implied, leave the score as not inferable rather than assigning zero.</p>
  </section>
}

function QualityRating({ packet, draft, updateDraft, onSelection, selectedSpan, spanComment, setSpanComment, spanTags, setSpanTags, addSpanComment }: {
  packet: Packet; draft: Draft; updateDraft: (patch: Partial<Draft>) => void
  onSelection: (event: React.SyntheticEvent<HTMLTextAreaElement>, component: string, index: number) => void
  selectedSpan: { component: string; index: number; start: number; end: number; quote: string } | null
  spanComment: string; setSpanComment: (value: string) => void; spanTags: string[]; setSpanTags: (value: string[]) => void; addSpanComment: () => void
}) {
  const components = ['strengths', 'weaknesses', 'suggestions'] as const
  return <section className="task-panel">
    {packet.task === 'assessment_quality' ? <>
      <div className="panel-title"><span>Assessment output</span><small>Select a score for each quality criterion</small></div>
      <div className="assessment-output">{packet.output?.dimensions?.map((dimension) => <article className="assessment-dimension" key={dimension.dimension_id}><header><strong>{dimension.dimension_id}</strong><span>Score {dimension.score.toFixed(1)}</span></header><p>{dimension.justification}</p><small>Evidence segments: {dimension.evidence_segment_ids.join(', ') || 'None cited'}</small></article>)}</div>
    </> : <>
      <div className="panel-title"><span>Feedback output</span><small>Select a text span to comment on it</small></div>
      <div className="feedback-output">{components.map((component) => <section className="feedback-component" key={component}><h3>{componentLabels[component]}</h3>{(packet.output?.[component] ?? []).length ? (packet.output?.[component] ?? []).map((item, index) => <div className="feedback-item" key={`${component}-${index}`}><textarea readOnly rows={Math.min(6, Math.max(2, Math.ceil(item.text.length / 90)))} value={item.text} onSelect={(event) => onSelection(event, component, index)} /><small>Evidence segments: {(item.evidence_segment_ids ?? []).join(', ') || 'None cited'}</small></div>) : <p className="empty-component">No items in this section.</p>}</section>)}</div>
      {selectedSpan && <div className="span-editor"><strong>Selected: “{selectedSpan.quote}”</strong><textarea value={spanComment} onChange={(event) => setSpanComment(event.target.value)} placeholder="Comment on this feedback span" rows={2} /><div className="tag-list">{TAGS.map((tag) => <label key={tag}><input type="checkbox" checked={spanTags.includes(tag)} onChange={(event) => setSpanTags(event.target.checked ? [...spanTags, tag] : spanTags.filter((value) => value !== tag))} />{tag}</label>)}</div><button className="icon-button" onClick={addSpanComment}>Add span comment</button></div>}
      {!!draft.spanComments.length && <div className="span-list"><h3>Span comments ({draft.spanComments.length})</h3>{draft.spanComments.map((comment, index) => <div key={`${comment.feedback_component}-${comment.feedback_item_index}-${comment.start_character}-${index}`}><span>{comment.feedback_component} · “{feedbackText(packet, comment.feedback_component, comment.feedback_item_index).slice(comment.start_character, comment.end_character)}”</span><p>{comment.comment}</p></div>)}</div>}
    </>}
    <div className="quality-criteria">{(packet.criteria ?? []).map((criterion) => {
      const answer = draft.criterionRatings[criterion.criterion_id]
      const scaleMin = packet.scale?.min ?? 1
      const scaleMax = packet.scale?.max ?? 5
      const scores = Array.from({ length: scaleMax - scaleMin + 1 }, (_, index) => scaleMin + index)
      const anchors = criterion.anchors ?? packet.scale?.anchors
      return <article className="criterion" key={criterion.criterion_id}>
        <div className="criterion-heading"><div><strong>{criterion.label}</strong><p>{criterion.description}</p></div><label className="unable"><input type="checkbox" checked={answer.unable_to_judge} onChange={(event) => updateDraft({ criterionRatings: { ...draft.criterionRatings, [criterion.criterion_id]: { ...answer, score: event.target.checked ? null : answer.score, unable_to_judge: event.target.checked } } })} />Unable to judge</label></div>
        <div className="score-buttons" role="group" aria-label={`${criterion.label} score`}>{scores.map((score) => <button key={score} title={anchors?.[String(score)]} className={answer.score === score && !answer.unable_to_judge ? 'chosen' : ''} disabled={answer.unable_to_judge} aria-pressed={answer.score === score && !answer.unable_to_judge} onClick={() => updateDraft({ criterionRatings: { ...draft.criterionRatings, [criterion.criterion_id]: { ...answer, score, unable_to_judge: false } } })}>{score}</button>)}</div>
        {anchors && <dl className="criterion-anchors">{scores.map((score) => <div key={score}><dt>{score}</dt><dd>{anchors[String(score)]}</dd></div>)}</dl>}
        <label className="field"><span>Rationale / comment <small>{answer.unable_to_judge ? 'Required' : 'Optional'}</small></span><textarea rows={2} value={answer.comment} onChange={(event) => updateDraft({ criterionRatings: { ...draft.criterionRatings, [criterion.criterion_id]: { ...answer, comment: event.target.value } } })} /></label>
      </article>
    })}</div>
    <label className="field overall-comment"><span>Overall comment <small>Optional</small></span><textarea rows={3} value={draft.overallComment} onChange={(event) => updateDraft({ overallComment: event.target.value })} /></label>
  </section>
}

const componentLabels: Record<string, string> = { strengths: 'Strengths', weaknesses: 'Development needs', suggestions: 'Next-step suggestions' }

function feedbackText(packet: Packet, component: string, index: number) {
  if (component !== 'strengths' && component !== 'weaknesses' && component !== 'suggestions') return ''
  return packet.output?.[component]?.[index]?.text ?? ''
}
