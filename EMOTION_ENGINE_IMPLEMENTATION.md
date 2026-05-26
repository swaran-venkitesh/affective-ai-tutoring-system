# Emotion Engine Implementation Notes

## Existing Code Reused
### From the tutor project
- `server.py`
  - preserved as the main tutoring engine
  - integrated the Emotion Engine as middleware, not a rewrite
- `camera_monitor.py`
  - kept as the existing VLM attention monitor
  - reused its `phone`, `away`, `distracted_side`, `sleepy`, and `multiple_people` signals as auxiliary context
- `LLM_QEWN_v2.py`
  - preserved as the tutor response generator
- `index.html`, `static/script.js`, `static/style.css`
  - preserved as the current UI shell
  - extended instead of replaced

### From prior emotion / camera work
- `emotion_tutor/src/detector.py`
- `emotion_tutor/src/emotion_model.py`
- `emotion_tutor/src/landmarks.py`
- `emotion_tutor/src/smoother.py`
- `emotion_tutor/src/mapper.py`
- `emotion_tutor/src/engagement.py`
- `emotion_tutor/src/utils.py`

These were copied into `emotion_engine/` and adapted to the tutor project.

## Newly Added
Package:
- `emotion_engine/`
  - `config.py`
  - `schemas.py`
  - `settings_store.py`
  - `text_affect.py`
  - `performance_affect.py`
  - `fusion.py`
  - `state_tracker.py`
  - `empathy_policy.py`
  - `llm_conditioning.py`
  - `speech_affect.py`
  - `camera_affect.py`
  - `engine.py`

Docs:
- `EMOTION_ENGINE_ARCHITECTURE.md`
- `EMOTION_ENGINE_IMPLEMENTATION.md`

## Files Modified
- `server.py`
- `templates/index.html`
- `static/script.js`
- `static/style.css`

## Backend Preservation Strategy
- existing tutor state machine stays in `server.py`
- existing `speak_chunks` flow stays unchanged
- existing board emission stays unchanged
- STT pause/resume stays unchanged
- current VLM camera monitor stays active
- Emotion Engine only contributes:
  - control packets
  - monitor data
  - auxiliary alert events

## Frontend Preservation Strategy
- current layout and sidebar stay intact
- new controls are placed inside the existing settings dropdown under `Advanced Settings`
- existing camera overlay remains the same base element
- face mesh overlay is drawn on a transparent canvas over the existing camera preview
- monitor popup is floating and hideable, not part of the main tutor layout

## Empathy Trigger Rules
### Triggered
- prolonged confusion
- non-productive confusion
- frustration after failure
- anxiety / self-doubt
- overload
- boredom / disengagement

### Suppressed
- productive confusion
- healthy progress / confidence
- moments where empathy would reduce clarity instead of helping

## Productive Confusion Treatment
Brief confusion is treated as normal learning effort when engagement remains healthy and harmful affect is not yet rising. In that case the policy avoids strong empathy and uses a small pedagogical assist such as a hint or one reduced step.

## Reactive Empathy
Reactive empathy is implemented as a policy output with an instructional action. The engine does not only mirror feelings; it pairs acknowledgment with a next step.

Examples of resulting actions:
- `simplify`
- `worked_example`
- `reassure_then_step`
- `reengage`

## Which Values Are Truly Live
### Truly live when camera is on
- face presence
- blink count / blink rate
- yawn count
- head / gaze direction heuristics
- raw face FER probabilities
- tutoring-face label
- valence / arousal
- engagement score
- VLM attention state when camera monitor is active

### Estimated
- fused tutoring affect state
- empathy decision
- confidence / overload / boredom / prolonged confusion state scores

### Camera required
- face mesh overlay
- blink / yawn / gaze values
- face FER values
- VLM camera context

### Text/performance only
- text affect
- performance affect
- control packet generation for tutoring turns

## Safe Fallback Behavior
- if Emotion Engine is OFF: tutor uses the old behavior
- if monitor is OFF: conditioning can still be active
- if face mesh overlay is OFF: internal camera affect can still continue
- if a model is missing: the engine logs and falls back safely

## Recommended Next Validation
- run the copied tutor live with the new toggles
- verify OFF mode matches prior tutor behavior
- verify monitor popup reflects live values only when available
- validate one short tutoring session with the engine ON
- then benchmark tutoring-state quality against annotated real tutor turns
