from __future__ import annotations

from .schemas import EmotionControlPacket, EmpathyDecision, FusedAffectState, StateConfidence, TrackedAffectState


class LLMConditioner:
    def build_packet(
        self,
        enabled: bool,
        tracked: TrackedAffectState,
        fused: FusedAffectState,
        decision: EmpathyDecision,
        state_confidence: StateConfidence,
        learner_state: dict,
        monitor_flags: dict[str, bool],
    ) -> EmotionControlPacket:
        return EmotionControlPacket(
            emotion_engine_enabled=enabled,
            affect_state=dict(tracked.scores),
            state_confidence=state_confidence,
            empathy_needed=decision.empathy_needed,
            empathy_type=decision.empathy_type,
            pedagogical_action=decision.pedagogical_action,
            tone_guidance=decision.tone_guidance,
            response_rules=list(decision.response_rules),
            policy_notes=decision.justification or decision.suppressed_reason,
            productive_confusion=tracked.productive_confusion,
            learner_state=dict(learner_state or {}),
            monitor_flags=dict(monitor_flags),
        )

    def render_prompt_block(self, packet: EmotionControlPacket) -> str:
        learner_state = dict(packet.learner_state or {})
        events = ", ".join(learner_state.get("detected_events") or []) or "none"
        rules = "\n".join(f"- {rule}" for rule in (packet.response_rules or []))
        if not rules:
            rules = "- teach clearly\n- keep empathy selective"

        lines = [
            "[ADAPTIVE TUTORING CONTEXT - internal only]",
            "Use this context only to adapt the teaching response. Do not mention the engine, policy, or hidden state.",
            "Use empathy only if the learner's emotional state is blocking learning. Keep empathy short, practical, and learning-focused.",
            "",
            "Learner state:",
            f"- confusion: {learner_state.get('confusion', 'low')}",
            f"- frustration: {learner_state.get('frustration', 'low')}",
            f"- engagement: {learner_state.get('engagement', 'low')}",
            f"- attention: {learner_state.get('attention', 'medium')}",
            f"- self_doubt: {str(bool(learner_state.get('self_doubt', False))).lower()}",
            f"- productive_confusion: {str(bool(packet.productive_confusion)).lower()}",
            f"- detected_events: {events}",
            "",
            "Recommended strategy:",
            f"- primary_action: {packet.pedagogical_action}",
            f"- tone: {packet.tone_guidance}",
            rules,
            "",
            "Response guardrails:",
            "- do not overpraise",
            "- do not repeat 'don't worry'",
            "- do not give away the full solution too early",
            "- keep the learner moving with one clear next step",
        ]
        return "\n".join(lines)
