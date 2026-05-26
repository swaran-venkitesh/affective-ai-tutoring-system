from __future__ import annotations

import json
from pathlib import Path

from local_secrets import load_local_env
from emotion_engine import EmotionEngine, EmotionEngineConfig


REPO_ROOT = Path(__file__).resolve().parent
load_local_env(REPO_ROOT)


def make_engine(enabled: bool) -> EmotionEngine:
    return EmotionEngine(
        EmotionEngineConfig(
            repo_root=REPO_ROOT,
            enabled_default=enabled,
            allow_client_toggle=False,
            show_monitor_ui=False,
            validation_log_enabled=False,
        ),
        log_callback=lambda _msg: None,
    )


def run_case(name: str, fn):
    result = fn()
    print(f"[PASS] {name}")
    return result


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def packet_summary(engine: EmotionEngine) -> dict:
    return {
        "packet": engine.latest_packet.to_dict(),
        "learner_state": engine.latest_learner_state.to_dict(),
        "validation": engine.current_validation_snapshot(),
        "prompt_block": engine.current_prompt_block(),
    }


def case_off_no_injection():
    engine = make_engine(enabled=False)
    packet = engine.handle_user_turn("I don't understand this step.", intent="not_understood", phase="QA", timestamp=1.0)
    summary = packet_summary(engine)
    assert_true(packet.emotion_engine_enabled is False, "OFF mode should disable emotion injection.")
    assert_true(packet.pedagogical_action == "normal_explain", "OFF mode should preserve normal explanation strategy.")
    assert_true(packet.policy_notes == "Emotion Engine disabled.", "OFF mode should record that the engine is disabled.")
    return summary


def case_first_time_confusion():
    engine = make_engine(enabled=True)
    packet = engine.handle_user_turn("I don't understand this step.", intent="not_understood", phase="QA", timestamp=1.0)
    summary = packet_summary(engine)
    assert_true(packet.emotion_engine_enabled is True, "ON mode should keep the engine enabled.")
    assert_true(summary["learner_state"]["productive_confusion"] is True, "A first confusion turn should be treated as productive confusion.")
    assert_true(packet.empathy_needed is False, "First-time confusion should not trigger empathy overuse.")
    assert_true(packet.pedagogical_action in {"clarify_step", "give_hint"}, "First-time confusion should request clarification or a hint.")
    return summary


def case_repeated_confusion():
    engine = make_engine(enabled=True)
    engine.handle_user_turn("I don't understand this step.", intent="not_understood", phase="QA", timestamp=1.0)
    packet = engine.handle_user_turn("I still don't get it. Please explain again.", intent="not_understood", phase="QA", timestamp=12.0)
    summary = packet_summary(engine)
    assert_true("repeated_confusion" in summary["learner_state"]["detected_events"], "Repeated confusion should be tracked as an event.")
    assert_true(packet.empathy_needed is True, "Repeated confusion should trigger adaptive support.")
    assert_true(packet.pedagogical_action in {"step_by_step", "simplify", "worked_example"}, "Repeated confusion should simplify the explanation.")
    return summary


def case_frustration():
    engine = make_engine(enabled=True)
    engine.record_qa_result(False, timestamp=1.0)
    engine.record_qa_result(False, timestamp=2.0)
    packet = engine.handle_user_turn("This is too hard. I'm frustrated.", intent="not_understood", phase="QA", timestamp=14.0)
    summary = packet_summary(engine)
    assert_true(packet.empathy_needed is True, "Frustration should trigger empathy.")
    assert_true(packet.pedagogical_action in {"worked_example", "simplify"}, "Frustration should reduce complexity.")
    return summary


def case_self_doubt():
    engine = make_engine(enabled=True)
    packet = engine.handle_user_turn("I'm dumb. I'll never understand this.", intent="not_understood", phase="QA", timestamp=20.0)
    summary = packet_summary(engine)
    assert_true(summary["learner_state"]["self_doubt"] is True, "Self-doubt should be detected.")
    assert_true(packet.empathy_needed is True, "Self-doubt should trigger support.")
    assert_true(packet.pedagogical_action == "reassure_then_step", "Self-doubt should use supportive reframing plus one next step.")
    return summary


def case_looking_away():
    engine = make_engine(enabled=True)
    engine.handle_support_event("looking_away", timestamp=30.0)
    packet = engine.handle_user_turn("Okay, continue.", intent="general", phase="QA", timestamp=31.0)
    summary = packet_summary(engine)
    assert_true("looking_away" in summary["learner_state"]["detected_events"], "Looking away should be tracked.")
    assert_true(packet.pedagogical_action == "reengage", "Looking away should trigger gentle re-engagement.")
    return summary


def case_sleepy():
    engine = make_engine(enabled=True)
    engine.handle_support_event("sleepy", timestamp=40.0)
    packet = engine.handle_user_turn("Continue.", intent="general", phase="QA", timestamp=41.0)
    summary = packet_summary(engine)
    assert_true("sleepy" in summary["learner_state"]["detected_events"], "Sleepy event should be tracked.")
    assert_true(packet.pedagogical_action == "gentle_break", "Sleepiness should suggest a brief reset.")
    return summary


def case_phone_usage():
    engine = make_engine(enabled=True)
    engine.handle_support_event("phone_detected", timestamp=50.0)
    packet = engine.handle_user_turn("Continue.", intent="general", phase="QA", timestamp=51.0)
    summary = packet_summary(engine)
    assert_true("phone_detected" in summary["learner_state"]["detected_events"], "Phone usage should be tracked.")
    assert_true(packet.pedagogical_action == "reengage", "Phone usage should trigger a short attention reminder.")
    return summary


def case_hand_raise():
    engine = make_engine(enabled=True)
    engine.handle_support_event("hand_raise", timestamp=60.0)
    packet = engine.handle_user_turn("Can you help me with this step?", intent="not_understood", phase="QA", timestamp=61.0)
    summary = packet_summary(engine)
    assert_true("hand_raise" in summary["learner_state"]["detected_events"], "Hand raise should be tracked.")
    assert_true(packet.pedagogical_action == "help_request", "Hand raise should be treated as a help request.")
    return summary


def case_camera_off_text_only():
    engine = make_engine(enabled=True)
    assert_true(engine.camera is None, "Camera runtime should not be required for text-only emotion support.")
    packet = engine.handle_user_turn("I still don't get it. This is too hard.", intent="not_understood", phase="QA", timestamp=65.0)
    summary = packet_summary(engine)
    assert_true(packet.emotion_engine_enabled is True, "Text-only mode should still use the emotion engine.")
    assert_true(packet.empathy_needed is True, "Text-only confusion/frustration should still trigger adaptive support.")
    assert_true(packet.pedagogical_action in {"step_by_step", "simplify", "worked_example"}, "Text-only path should still adapt the strategy.")
    assert_true(summary["learner_state"]["attention_status"] in {"unknown", "reflecting", "Camera off"}, "Text-only path should not depend on camera attention.")
    return summary


def case_engaged_learner():
    engine = make_engine(enabled=True)
    packet = engine.handle_user_turn("Okay, I understand. Give me a harder question.", intent="general", phase="QA", timestamp=70.0)
    summary = packet_summary(engine)
    assert_true(packet.empathy_needed is False, "Engaged learners should not receive unnecessary empathy.")
    assert_true(packet.pedagogical_action == "challenge_continue", "Engaged learners should be allowed to continue with challenge.")
    return summary


def main() -> None:
    cases = [
        ("OFF mode keeps tutoring normal", case_off_no_injection),
        ("ON mode first confusion stays productive", case_first_time_confusion),
        ("ON mode repeated confusion escalates support", case_repeated_confusion),
        ("ON mode frustration reduces complexity", case_frustration),
        ("ON mode self-doubt triggers reframing", case_self_doubt),
        ("ON mode looking away re-engages gently", case_looking_away),
        ("ON mode sleepy suggests reset", case_sleepy),
        ("ON mode phone usage reminds briefly", case_phone_usage),
        ("ON mode hand raise becomes help request", case_hand_raise),
        ("ON mode camera off still works from text only", case_camera_off_text_only),
        ("ON mode engaged learner continues normally", case_engaged_learner),
    ]

    outputs = {}
    for name, fn in cases:
        outputs[name] = run_case(name, fn)

    print("\nSample OFF vs ON prompt evidence:")
    print(json.dumps(
        {
            "off_packet": outputs["OFF mode keeps tutoring normal"]["packet"],
            "on_first_confusion_packet": outputs["ON mode first confusion stays productive"]["packet"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
