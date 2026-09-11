"""
Unit tests for the Part B decision layer.

These deliberately import only app.reasoning, which pulls in nothing heavier
than the standard library. The routing and guardrail logic is therefore
testable without torch, without a checkpoint, and without an API key - which
is the point: the decision layer is our code, so it should be verifiable on
its own terms rather than only observable through the model.

Run:  python -m pytest tests/ -v
  or: python tests/test_reasoning.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import reasoning


IMAGE = "sample_images/damaged_parcel.jpg"


def det(label, confidence, bbox=None):
    return {"label": label, "confidence": confidence, "bbox": bbox or [0.0, 0.0, 10.0, 10.0]}


class TestIntentRouting(unittest.TestCase):
    """The router must justify every decision, and must not call the detector
    for questions the detector cannot answer."""

    def test_damage_question_routes_to_vision(self):
        route, why = reasoning.route_intent(
            "Is there any damage to this parcel?", IMAGE)
        self.assertEqual(route, reasoning.ROUTE_VISION)
        self.assertIn("damage", why)

    def test_counting_packages_routes_to_vision(self):
        route, _ = reasoning.route_intent("How many parcels are in this image?", IMAGE)
        self.assertEqual(route, reasoning.ROUTE_VISION)

    def test_damage_question_with_image_routes_to_vision(self):
        route, _ = reasoning.route_intent("Is this carton crushed or torn?", IMAGE)
        self.assertEqual(route, reasoning.ROUTE_VISION)

    def test_carrier_history_skips_the_detector(self):
        """Custody history lives in the ledger. Pixels cannot answer it, so
        spending an inference on it would be pure waste."""
        route, _ = reasoning.route_intent("Which carrier handled this in transit?", IMAGE)
        self.assertEqual(route, reasoning.ROUTE_LEDGER)

    def test_unrelated_question_is_out_of_scope(self):
        route, _ = reasoning.route_intent("What is the weather in Chennai today?", IMAGE)
        self.assertEqual(route, reasoning.ROUTE_OUT_OF_SCOPE)

    def test_visual_question_without_image_falls_back(self):
        """Visual intent with no image must not pretend to inspect anything."""
        route, why = reasoning.route_intent("Is the carton damaged?", None)
        self.assertEqual(route, reasoning.ROUTE_LEDGER)
        self.assertIn("no image_path", why)

    def test_word_boundary_prevents_false_unsupported(self):
        """'sealant' contains 'seal', but this is not a question about a seal.
        Substring matching would misfire the UNSUPPORTED_CAPABILITY branch."""
        route, _ = reasoning.route_intent("Who is our sealant supplier?", IMAGE)
        self.assertNotEqual(route, reasoning.ROUTE_UNSUPPORTED)

    def test_procurement_question_is_out_of_scope(self):
        """Commercial questions must not reach the detector even with an image
        attached, or the LLM answers a supplier question from parcel boxes."""
        route, _ = reasoning.route_intent("Who is our sealant supplier?", IMAGE)
        self.assertEqual(route, reasoning.ROUTE_OUT_OF_SCOPE)


class TestRouterDoesNotOverRefuse(unittest.TestCase):
    """A keyword whitelist cannot enumerate every phrasing, so "no keyword
    matched" must never by itself cause a refusal when an image was supplied.

    Before this was fixed, 8 of these 16 questions returned OUT_OF_SCOPE,
    including the brief's own example "What's the most common object here?".
    """

    NATURAL_QUESTIONS = [
        "What is the most common object here?",   # verbatim from the brief
        "What objects do you see?",
        "How many objects are in this image?",
        "Describe this image",
        "What is in this picture?",
        "Count the boxes",
        "Is the package okay?",
        "Should I accept this delivery?",
        "Any issues with this shipment?",
        "Is this parcel fine to route?",
        "What did you detect?",
        "Summarize the findings",
        "Is there a problem?",
        "Can this be auto-routed?",
        "How many damaged packages?",
        "What class did you find?",
    ]

    def test_natural_questions_reach_the_detector(self):
        refused = [q for q in self.NATURAL_QUESTIONS
                   if reasoning.route_intent(q, IMAGE)[0] == reasoning.ROUTE_OUT_OF_SCOPE]
        self.assertEqual(refused, [], "these should inspect the image, not refuse")

    def test_unknown_phrasing_with_an_image_inspects_it(self):
        """Positive evidence is required to refuse, not an absent keyword."""
        route, why = reasoning.route_intent("Give me your assessment", IMAGE)
        self.assertEqual(route, reasoning.ROUTE_VISION)
        self.assertIn("an image was supplied", why)

    def test_unknown_phrasing_without_an_image_still_refuses(self):
        """With no image there is nothing to inspect, so refusal is correct."""
        route, _ = reasoning.route_intent("Give me your assessment", None)
        self.assertEqual(route, reasoning.ROUTE_OUT_OF_SCOPE)


class TestUnsupportedCapability(unittest.TestCase):
    """Questions about things outside the trained label set must be refused as
    a capability limit, NOT answered from whatever the detector happens to see,
    and NOT conflated with low-confidence evidence."""

    def test_person_question_is_refused_not_detected(self):
        """The classic example from the brief. This model has no person class,
        so answering it from package boxes would be a confident wrong answer."""
        route, why = reasoning.route_intent("How many people are in this image?", IMAGE)
        self.assertEqual(route, reasoning.ROUTE_UNSUPPORTED)
        self.assertIn("no person class", why)

    def test_label_question_is_refused(self):
        """The printed-label class was cut after the data audit, so questions
        about label legibility must refuse rather than answer from parcel boxes."""
        route, why = reasoning.route_intent("Is the shipping label readable?", IMAGE)
        self.assertEqual(route, reasoning.ROUTE_UNSUPPORTED)
        self.assertIn("localises parcels", why)

    def test_vocabularies_are_disjoint(self):
        """A token in both sets is dead code: the unsupported check runs first,
        so the vision entry could never fire. Silent, and easy to reintroduce."""
        self.assertEqual(reasoning.VISION_TOKENS & reasoning.UNSUPPORTED_TOKENS, set())

    def test_every_unsupported_token_has_a_reason(self):
        """route_intent indexes UNSUPPORTED_REASONS directly, so a token
        without a reason raises KeyError on a live request."""
        self.assertEqual(reasoning.UNSUPPORTED_TOKENS - set(reasoning.UNSUPPORTED_REASONS), set())

    def test_seal_question_is_refused(self):
        route, why = reasoning.route_intent("Has the security seal been tampered with?", IMAGE)
        self.assertEqual(route, reasoning.ROUTE_UNSUPPORTED)
        self.assertIn("too few training examples", why)

    def test_thermal_question_is_refused(self):
        route, why = reasoning.route_intent("Is there a thermal hotspot on this parcel?", IMAGE)
        self.assertEqual(route, reasoning.ROUTE_UNSUPPORTED)
        self.assertIn("thermal imaging", why)

    def test_unsupported_beats_generic_visual_match(self):
        """'Is the seal on this box intact' hits `box` and `intact` as visual
        tokens. The unsupported check must win, or the detector gets called and
        answers a seal question using package boxes."""
        route, _ = reasoning.route_intent("Is the seal on this box intact?", IMAGE)
        self.assertEqual(route, reasoning.ROUTE_UNSUPPORTED)


class TestConfidenceGuardrail(unittest.TestCase):
    """The guardrail is the honesty mechanism. It must refuse on weak evidence
    and must not refuse on strong evidence."""

    def test_weak_defect_detection_halts(self):
        detections = [det("damaged-package", 0.47)]
        passed, why = reasoning.evaluate_guardrail(detections, 0.47)
        self.assertFalse(passed)
        self.assertIn("0.47", why)

    def test_confidence_just_below_threshold_halts(self):
        passed, _ = reasoning.evaluate_guardrail([det("damaged-package", 0.6499)], 0.6499)
        self.assertFalse(passed)

    def test_confidence_at_threshold_passes(self):
        passed, _ = reasoning.evaluate_guardrail([det("damaged-package", 0.65)], 0.65)
        self.assertTrue(passed)

    def test_strong_defect_passes(self):
        passed, why = reasoning.evaluate_guardrail([det("damaged-package", 0.88)], 0.88)
        self.assertTrue(passed)
        self.assertIn("clears threshold", why)

    def test_clean_parcel_confidently_located_is_clear(self):
        """No defect, but the parcel itself was found. A negative finding is
        supportable, so this must NOT be reported as insufficient."""
        detections = [det("package", 0.94)]
        passed, why = reasoning.evaluate_guardrail(detections, 0.0)
        self.assertTrue(passed)
        self.assertIn("negative finding is supportable", why)

    def test_empty_frame_cannot_support_a_negative(self):
        """Nothing detected at all. We cannot claim the parcel is fine when we
        cannot even see a parcel."""
        passed, why = reasoning.evaluate_guardrail([], 0.0)
        self.assertFalse(passed)
        self.assertIn("cannot support a negative finding", why)

    def test_low_confidence_package_cannot_clear_a_parcel(self):
        """A parcel was seen, but only weakly. That is not a firm enough
        localisation to assert the parcel is undamaged."""
        passed, _ = reasoning.evaluate_guardrail([det("package", 0.41)], 0.0)
        self.assertFalse(passed)

    def test_weak_defect_outranks_a_confident_container(self):
        """A clearly visible box does not license ignoring an ambiguous tear."""
        detections = [det("package", 0.96), det("damaged-package", 0.51)]
        passed, _ = reasoning.evaluate_guardrail(detections, 0.51)
        self.assertFalse(passed)


class TestPromptAssembly(unittest.TestCase):
    """The LLM must receive the structured output verbatim, never a paraphrase."""

    def test_prompt_carries_detections_and_ledger(self):
        detections = [det("damaged-package", 0.88)]
        ledger = {"carrier": "Apex Logistics", "origin_label_status": "INTACT"}
        prompt = reasoning._build_user_prompt("Is this a claim?", detections,
                                              {"damaged-package": 1}, ledger)
        self.assertIn("0.88", prompt)
        self.assertIn("Apex Logistics", prompt)
        self.assertIn("INTACT", prompt)

    def test_missing_ledger_is_stated_not_hidden(self):
        prompt = reasoning._build_user_prompt("Is this a claim?", [], {}, None)
        self.assertIn("no record found", prompt)


class TestDetectionDeduplication(unittest.TestCase):
    """RT-DETR is NMS-free, so two decoder queries can lock onto one parcel and
    both survive. Measured on the v3 test split: 12.4% of images emitted more
    boxes than there were objects, and suppressing the weaker of two
    heavily-overlapping same-class boxes removed 90 of 420 false positives at
    zero recall cost. These pin the behaviour so it cannot silently regress.
    """

    def test_real_duplicate_pair_collapses(self):
        """The exact pair found during v3 failure analysis."""
        from app.detector import deduplicate
        dets = [
            {"label": "package", "confidence": 0.959, "bbox": [210.2, 200.7, 271.6, 301.5]},
            {"label": "package", "confidence": 0.575, "bbox": [210.2, 200.5, 271.7, 301.5]},
        ]
        kept = deduplicate(dets)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["confidence"], 0.959, "must keep the stronger box")

    def test_different_classes_are_never_merged(self):
        """A damaged parcel overlapping an intact one is a real observation."""
        from app.detector import deduplicate
        dets = [
            {"label": "package", "confidence": 0.9, "bbox": [0, 0, 100, 100]},
            {"label": "damaged-package", "confidence": 0.8, "bbox": [0, 0, 100, 100]},
        ]
        self.assertEqual(len(deduplicate(dets)), 2)

    def test_separate_parcels_both_survive(self):
        """Two distinct parcels of the same class must not collapse."""
        from app.detector import deduplicate
        dets = [
            {"label": "package", "confidence": 0.9, "bbox": [0, 0, 100, 100]},
            {"label": "package", "confidence": 0.8, "bbox": [200, 200, 300, 300]},
        ]
        self.assertEqual(len(deduplicate(dets)), 2)

    def test_partial_overlap_below_threshold_survives(self):
        """Touching parcels on a conveyor overlap without being duplicates."""
        from app.detector import deduplicate
        dets = [
            {"label": "package", "confidence": 0.9, "bbox": [0, 0, 100, 100]},
            {"label": "package", "confidence": 0.8, "bbox": [60, 0, 160, 100]},
        ]
        self.assertEqual(len(deduplicate(dets)), 2)

if __name__ == "__main__":
    unittest.main(verbosity=2)


