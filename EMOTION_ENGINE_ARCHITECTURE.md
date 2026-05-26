# Emotion Engine Architecture

## Goal
Add an optional, text-first but multimodal-capable Emotion Engine to the tutor without replacing the current tutor, board sync, STT/TTS flow, or camera monitor.

## Pipeline
student input
-> routing and tutoring context
-> text affect estimation
-> performance-affect estimation
-> optional camera / face / VLM auxiliary signals
-> multimodal fusion
-> affect state tracking over time
-> empathy policy decision
-> LLM control packet
-> existing tutor generation
-> existing board / speech flow unchanged

## Target Learner States
- `confusion`
- `prolonged_confusion`
- `frustration`
- `boredom`
- `anxiety_self_doubt`
- `engagement`
- `confidence`
- `overload`

These are tutoring-oriented learner states, not claims of true biological emotion.

## Modality Priority
1. Text affect
2. Performance-derived affect
3. Camera/VLM attention signals
4. Face FER + mesh geometry
5. Speech affect placeholder

Text and tutoring-performance evidence are intentionally weighted above camera and face signals. Camera/face are auxiliary unless corroborated by stronger evidence.

## Model Choices
### Text affect
- Primary design target: `SamLowe/roberta-base-go_emotions`
- Reason: dedicated RoBERTa-family affect classifier, better fit for tutoring affect than using the tutor LLM alone
- Runtime behavior: loads from local cache if available, otherwise falls back to deterministic tutoring-oriented heuristics so the tutor does not break offline

### Face affect
- Primary local backend: `EmoNet`
- Fallback backend: `EmotiEffLib enet_b0_8_va_mtl`
- Reason: direct valence/arousal support plus practical webcam deployment

### Face mesh / landmarks
- `MediaPipe Face Mesh`
- Used for face presence, blink count/rate, prolonged closure, yawn-like events, gaze/head direction, and optional overlay

### Camera/VLM monitor
- Existing `camera_monitor.py` remains the VLM attention layer
- Outputs such as `phone`, `away`, `distracted_side`, `sleepy`, and `multiple_people` are fed into the Emotion Engine as auxiliary context

## Selective Empathy Policy
Empathy is not a constant tone. It is a policy decision.

### Triggered when affect is likely harming learning
- repeated failure with frustration
- anxiety or self-doubt
- overload
- boredom / disengagement
- confusion that persists long enough to stop productive progress

### Intentionally suppressed
- brief confusion with good engagement
- productive struggle that has not turned into frustration or overload
- strong progress / confidence states
- assessment moments where excessive comfort would reduce clarity

### Productive confusion
The engine treats confusion as productive when confusion is present but:
- it is still short-lived
- engagement is still healthy
- frustration and anxiety remain low

In that case the policy avoids strong emotional language and prefers one hint or one smaller step.

### Reactive empathy vs mirroring
The policy prefers reactive empathy over simple emotional mirroring.

Examples:
- weaker: "I understand you feel frustrated."
- stronger: "This is getting frustrating. Let's reduce it to one smaller step."

The engine therefore outputs short acknowledgment plus a pedagogical action such as:
- `normal_explain`
- `simplify`
- `give_hint`
- `worked_example`
- `recap_prerequisite`
- `reassure_then_step`
- `reengage`
- `challenge_continue`

## Observability
The frontend monitor separates:
- raw modality states
- fused tutor state
- final policy output

When available it also shows:
- raw face emotion probabilities
- tutoring-face label
- valence / arousal
- attention status
- engagement
- blink / yawn / face presence

Unavailable values are intentionally left blank or shown as `Not available`.

## Safe Fallback
- If Emotion Engine is OFF, current tutor behavior remains unchanged.
- If a model is unavailable, the engine falls back safely and keeps the tutor running.
- Face mesh overlay only controls rendering; it does not require the entire tutor flow to change.

## Metrics and Evaluation Note
The engine should be evaluated by modality and by tutoring-state usefulness, not by a single headline number.

Recommended benchmark directions:
- Text affect: GoEmotions-style held-out text evaluation
- Face FER: AffectNet / RAF-DB style FER evaluation
- Tutoring states such as boredom / engagement / confusion / frustration: DAiSEE-style evaluation
- End-to-end tutoring policy: turn-level annotation on real tutor sessions

For live UI, showing the full raw face probability distribution is preferable to showing only one face label.
