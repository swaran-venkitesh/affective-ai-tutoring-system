from __future__ import annotations

from .config import EmotionEngineConfig
from .schemas import EmpathyDecision, ModalityEstimate, TrackedAffectState


class EmpathyPolicyEngine:
    def __init__(self, config: EmotionEngineConfig) -> None:
        self.config = config

    def decide(
        self,
        tracked: TrackedAffectState,
        performance: ModalityEstimate | None = None,
        active_events: list[str] | None = None,
    ) -> EmpathyDecision:
        scores = tracked.scores
        perf_details = performance.details if performance else {}
        incorrect_streak = int(perf_details.get("incorrect_streak", 0))
        clarification_streak = int(perf_details.get("clarification_streak", 0))
        perf_evidence = {str(item) for item in (perf_details.get("evidence") or [])}
        events = {str(item) for item in (active_events or [])}

        if "hand_raise" in events:
            return EmpathyDecision(
                empathy_needed=False,
                empathy_type="none",
                pedagogical_action="help_request",
                tone_guidance="attentive, calm, concise",
                response_rules=[
                    "treat the raised hand as a help request",
                    "ask what specific step needs help",
                    "do not give a long answer before clarifying the blockage",
                ],
                justification="The learner explicitly signaled a request for help.",
            )

        if "sleepy" in events:
            return EmpathyDecision(
                empathy_needed=True,
                empathy_type="light",
                pedagogical_action="gentle_break",
                tone_guidance="gentle, brief, respectful",
                response_rules=[
                    "suggest a short reset or break",
                    "do not scold or repeat the warning aggressively",
                    "offer a short recap when the learner returns",
                ],
                justification="Sleepiness is likely blocking attention and retention.",
            )

        if "phone_detected" in events or "looking_away" in events:
            return EmpathyDecision(
                empathy_needed=True,
                empathy_type="light",
                pedagogical_action="reengage",
                tone_guidance="brief, calm, purposeful",
                response_rules=[
                    "use one short attention reminder",
                    "invite the learner back with a small next step",
                    "avoid repeated nagging",
                ],
                justification="Attention drift is reducing effective learning.",
            )

        if tracked.productive_confusion and clarification_streak <= 1 and "repeated_confusion" not in events:
            return EmpathyDecision(
                empathy_needed=False,
                empathy_type="none",
                pedagogical_action="clarify_step",
                tone_guidance="clear, calm, confidence-building",
                response_rules=[
                    "do not over-comfort",
                    "treat confusion as active thinking",
                    "offer one hint or one smaller step",
                ],
                justification="Confusion is present but still productive.",
                suppressed_reason="productive_confusion",
            )

        if scores["anxiety_self_doubt"] >= 0.56:
            return EmpathyDecision(
                empathy_needed=True,
                empathy_type="reactive",
                pedagogical_action="reassure_then_step",
                tone_guidance="warm, calm, confidence-building",
                response_rules=[
                    "briefly reframe the self-judgment",
                    "avoid overpraising",
                    "shrink the task to one small achievable step",
                    "keep learning progress primary",
                ],
                justification="Self-doubt is likely to reduce persistence.",
            )

        if scores["frustration"] >= 0.60 and (incorrect_streak >= 1 or clarification_streak >= 2):
            return EmpathyDecision(
                empathy_needed=True,
                empathy_type="reactive",
                pedagogical_action="worked_example",
                tone_guidance="warm, steady, problem-solving",
                response_rules=[
                    "acknowledge the difficulty briefly",
                    "switch to a simpler scaffold or worked example",
                    "reduce complexity immediately",
                    "end with one concrete next step",
                ],
                justification="Frustration is now blocking progress and needs reactive support.",
            )

        if scores["prolonged_confusion"] >= 0.58 or (
            scores["confusion"] >= 0.60 and tracked.durations_sec["confusion"] >= self.config.prolonged_confusion_seconds
        ):
            return EmpathyDecision(
                empathy_needed=True,
                empathy_type="light",
                pedagogical_action="step_by_step",
                tone_guidance="warm, calm, reduce cognitive load",
                response_rules=[
                    "acknowledge briefly",
                    "normalize the difficulty without drama",
                    "break the explanation into step-by-step chunks",
                    "end with one actionable next move",
                ],
                justification="Confusion has lasted long enough to impair learning.",
            )

        if "repeated_confusion" in events or clarification_streak >= 2:
            return EmpathyDecision(
                empathy_needed=True,
                empathy_type="light",
                pedagogical_action="step_by_step",
                tone_guidance="warm, calm, reduce cognitive load",
                response_rules=[
                    "acknowledge briefly",
                    "simplify the explanation",
                    "break the solution into step-by-step chunks",
                    "give one concrete next step",
                ],
                justification="Repeated confusion now warrants adaptive support.",
            )

        if scores["confusion"] >= 0.50 and scores["frustration"] >= 0.18:
            return EmpathyDecision(
                empathy_needed=True,
                empathy_type="light",
                pedagogical_action="simplify",
                tone_guidance="warm, calm, reduce cognitive load",
                response_rules=[
                    "acknowledge briefly",
                    "treat the confusion as turning non-productive",
                    "simplify the explanation immediately",
                    "give one small next step",
                ],
                justification="Confusion is now mixing with frustration and needs support.",
            )

        if clarification_streak == 1 and ("clarify_request" in perf_evidence or "hint_request" in perf_evidence):
            return EmpathyDecision(
                empathy_needed=False,
                empathy_type="none",
                pedagogical_action="clarify_step" if "clarify_request" in perf_evidence else "give_hint",
                tone_guidance="clear, calm, learning-focused",
                response_rules=[
                    "keep empathy minimal",
                    "clarify only the blocked step",
                    "offer one hint or one reduced step",
                    "avoid turning a productive struggle into over-comforting",
                ],
                justification="The learner is engaged and asking for a first clarification.",
            )

        if scores["overload"] >= 0.58 or clarification_streak >= 3:
            return EmpathyDecision(
                empathy_needed=True,
                empathy_type="reactive",
                pedagogical_action="simplify",
                tone_guidance="calm, slow, structured",
                response_rules=[
                    "reduce load immediately",
                    "do not dump more information",
                    "use one concise step or recap a prerequisite",
                ],
                justification="Cognitive load appears too high for the current pace.",
            )

        if scores["boredom"] >= 0.58 or (scores["engagement"] <= 0.30 and scores["confusion"] < 0.42 and scores["frustration"] < 0.42):
            return EmpathyDecision(
                empathy_needed=True,
                empathy_type="light",
                pedagogical_action="reengage",
                tone_guidance="brief, energizing, purposeful",
                response_rules=[
                    "do not use sentimental language",
                    "re-engage with a question, recap, or choice",
                    "offer a hint, example, or quick check",
                ],
                justification="Engagement is dropping and re-engagement is needed.",
            )

        if scores["confusion"] >= 0.48 and scores["frustration"] < 0.48 and scores["anxiety_self_doubt"] < 0.48:
            return EmpathyDecision(
                empathy_needed=False,
                empathy_type="none",
                pedagogical_action="clarify_step",
                tone_guidance="clear, calm, concise",
                response_rules=[
                    "treat the confusion as normal effort",
                    "clarify the blocked step briefly",
                    "keep the learner thinking",
                ],
                justification="Confusion is present, but it does not yet require empathy.",
            )

        if scores["confusion"] >= 0.48 or scores["frustration"] >= 0.48:
            return EmpathyDecision(
                empathy_needed=True,
                empathy_type="light" if scores["confusion"] >= scores["frustration"] else "reactive",
                pedagogical_action="simplify",
                tone_guidance="warm, calm, learning-focused",
                response_rules=[
                    "acknowledge briefly",
                    "do not mirror emotion for too long",
                    "offer one practical next step",
                    "keep teaching value higher than comfort language",
                ],
                justification="Confusion or frustration is starting to interfere with learning.",
            )

        if scores["confidence"] >= 0.55 and scores["engagement"] >= 0.50:
            return EmpathyDecision(
                empathy_needed=False,
                empathy_type="none",
                pedagogical_action="challenge_continue",
                tone_guidance="clear, upbeat, concise",
                response_rules=[
                    "keep empathy minimal",
                    "maintain instructional momentum",
                    "offer the next challenge cleanly",
                ],
                justification="The learner appears stable and ready to continue.",
            )

        return EmpathyDecision(
            empathy_needed=False,
            empathy_type="none",
            pedagogical_action="normal_explain",
            tone_guidance="clear, calm, concise",
            response_rules=[
                "teach clearly",
                "keep empathy selective",
                "do not overdramatize normal effort",
            ],
            justification="No harmful affect signal is strong enough to intervene.",
        )
