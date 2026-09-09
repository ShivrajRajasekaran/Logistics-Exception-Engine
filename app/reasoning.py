"""
Part B - hand-written reasoning layer.

NO AGENTIC FRAMEWORK IS USED OR IMPORTED HERE.
No LangChain, no LlamaIndex, no CrewAI, no AutoGen, no Haystack, no Semantic
Kernel. The control flow below is ordinary Python `if` statements, and the
single LLM call goes through the official OpenAI SDK directly.

Pipeline: route intent -> (maybe) detect -> guardrail -> ledger lookup -> synthesize.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("exception-engine.reasoning")

GUARDRAIL_THRESHOLD = float(os.getenv("GUARDRAIL_THRESHOLD", "0.65"))

# When true, an image with zero critical-class detections also halts as
# INSUFFICIENT_INFORMATION. Default false: a legible image with no defect is a
# genuine CLEAR result, and reporting it as "insufficient" would be dishonest.
STRICT_ZERO_DETECTION_HALT = os.getenv("STRICT_ZERO_DETECTION_HALT", "false").lower() == "true"

# --- Intent routing vocabulary -------------------------------------------
# Deterministic keyword routing, deliberately chosen over an LLM classifier:
# the router must be explainable line-by-line and must not cost a network call
# just to decide whether a network call is needed.

VISION_TOKENS = {
    "damage", "damaged", "dent", "dented", "torn", "tear", "ripped", "crushed",
    "opened", "punctured", "wet", "collapsed",
    "intact", "condition", "inspect", "inspection", "visible", "see", "look", "image",
    "photo", "picture", "count", "how many", "box", "carton", "parcel", "package",
    "defect", "anomaly", "claim", "liability",
}
# Note what is NOT here: label, barcode, seal, tape, tamper. Each names a
# concept the trained model cannot see, so each lives in UNSUPPORTED_TOKENS
# below. A token must never appear in both sets - the unsupported check runs
# first, so a duplicate would silently make the vision entry dead code.

# Questions that ARE about the image but name something this model cannot
# detect. These get an explicit capability refusal rather than a detector call.
#
# Routing them to the detector would be the worst outcome: it would return
# packages, and the LLM would then answer a question about people using boxes.
# Saying "I was not trained for that" is the honest response, and it is a
# different failure from "I looked and could not tell".
UNSUPPORTED_TOKENS = {
    "person", "people", "worker", "operator", "staff", "anyone", "human",
    "label", "labels", "barcode", "printed", "readable", "legible", "address",
    "thermal", "heat", "temperature", "hotspot", "overheat", "cold",
    "seal", "tape", "tamper", "tampered", "sealed",
}

# Why each is unsupported, surfaced to the caller so the limit is legible.
UNSUPPORTED_REASONS = {
    "label": "the printed-label class was cut; the model localises parcels, not markings",
    "labels": "the printed-label class was cut; the model localises parcels, not markings",
    "barcode": "the printed-label class was cut; the model localises parcels, not markings",
    "printed": "the printed-label class was cut; the model localises parcels, not markings",
    "readable": "the printed-label class was cut; the model localises parcels, not markings",
    "legible": "the printed-label class was cut; the model localises parcels, not markings",
    "address": "the printed-label class was cut; the model localises parcels, not markings",
    "person": "the model has no person class",
    "people": "the model has no person class",
    "worker": "the model has no person class",
    "operator": "the model has no person class",
    "staff": "the model has no person class",
    "anyone": "the model has no person class",
    "human": "the model has no person class",
    "thermal": "thermal imaging was cut from this model",
    "heat": "thermal imaging was cut from this model",
    "temperature": "thermal imaging was cut from this model",
    "hotspot": "thermal imaging was cut from this model",
    "overheat": "thermal imaging was cut from this model",
    "cold": "thermal imaging was cut from this model",
    "seal": "seal tampering had too few training examples to support a class",
    "tape": "seal tampering had too few training examples to support a class",
    "tamper": "seal tampering had too few training examples to support a class",
    "tampered": "seal tampering had too few training examples to support a class",
    "sealed": "seal tampering had too few training examples to support a class",
}

ROUTE_UNSUPPORTED = "UNSUPPORTED_CAPABILITY"

LEDGER_TOKENS = {
    "carrier", "origin", "dispatched", "manifest", "transit", "history", "hub",
    "sku", "shipped", "who handled", "when did", "route", "ledger", "record",
}

# Questions that touch neither the image nor the parcel record.
OUT_OF_SCOPE_TOKENS = {
    "weather", "stock price", "capital of", "translate", "joke", "recipe",
    "who are you", "your name", "sla definition", "company policy", "holiday",
}

ROUTE_VISION = "VISION_REQUIRED"
ROUTE_LEDGER = "LEDGER_ONLY"
ROUTE_OUT_OF_SCOPE = "OUT_OF_SCOPE"


def _normalize(query: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", query.lower())


def _hits(text: str, vocabulary: set) -> List[str]:
    """Phrase-aware match: multi-word tokens use substring, single words use
    word-boundary matching so 'seal' does not fire on 'sealant supplier'."""
    found = []
    words = set(text.split())
    for token in vocabulary:
        if " " in token:
            if token in text:
                found.append(token)
        elif token in words:
            found.append(token)
    return found


def route_intent(query: str, image_path: Optional[str]) -> Tuple[str, str]:
    """Decide whether this question needs the detector. Returns (route, why)."""
    text = _normalize(query)

    out_of_scope = _hits(text, OUT_OF_SCOPE_TOKENS)
    vision = _hits(text, VISION_TOKENS)
    ledger = _hits(text, LEDGER_TOKENS)
    unsupported = _hits(text, UNSUPPORTED_TOKENS)

    # Rule 0: the question is visual but names something outside this model's
    # label set. Checked FIRST, ahead of the generic visual match, because
    # "is the seal on this box intact" hits `box` and `intact` too - and
    # answering it from package detections would be a confident wrong answer.
    if unsupported:
        reasons = sorted({UNSUPPORTED_REASONS[t] for t in unsupported})
        return ROUTE_UNSUPPORTED, "asks about %s; %s" % (sorted(unsupported), "; ".join(reasons))

    # Rule 1: clearly unrelated to this parcel, and nothing visual asked.
    if out_of_scope and not vision:
        return ROUTE_OUT_OF_SCOPE, "matched out-of-scope terms %s with no visual intent" % sorted(out_of_scope)

    # Rule 2: no visual vocabulary at all -> the pixels cannot help.
    if not vision:
        if ledger:
            return ROUTE_LEDGER, "matched record-only terms %s; parcel history answers this" % sorted(ledger)
        return ROUTE_OUT_OF_SCOPE, "no visual or record vocabulary matched"

    # Rule 3: visual intent but no image supplied -> cannot run the detector.
    if not image_path:
        return ROUTE_LEDGER, "visual intent detected but no image_path supplied; falling back to record"

    return ROUTE_VISION, "matched visual terms %s" % sorted(vision)


def evaluate_guardrail(detections: List[dict], max_critical: float) -> Tuple[bool, str]:
    """
    Confidence guardrail. Returns (passed, explanation).

    Hard rule: if a defect class was detected but its confidence sits below
    GUARDRAIL_THRESHOLD, we stop before the LLM. A weak defect signal is the
    exact situation where a language model would confabulate a claim.
    """
    has_critical = max_critical > 0.0

    if has_critical and max_critical < GUARDRAIL_THRESHOLD:
        return False, (
            "Defect signal present but weak: peak critical-class confidence "
            "%.2f is below the operational threshold %.2f. Refusing to assign "
            "liability from an ambiguous detection." % (max_critical, GUARDRAIL_THRESHOLD)
        )

    if not has_critical:
        if STRICT_ZERO_DETECTION_HALT:
            return False, "No critical-class detection reached the confidence threshold."
        # Trust a negative only if the parcel itself was located confidently.
        # Otherwise the frame may be blank, dark, or badly framed.
        from app.detector import ANCHOR_CLASS

        container = [d for d in detections
                     if d["label"] == ANCHOR_CLASS and d["confidence"] >= GUARDRAIL_THRESHOLD]
        if not container:
            return False, (
                "No defect detected, and no parcel was localized above %.2f either. "
                "The frame cannot support a negative finding - it may be empty, "
                "occluded, or out of focus." % GUARDRAIL_THRESHOLD
            )
        return True, (
            "Parcel localized at confidence %.2f with no defect class above "
            "threshold. A negative finding is supportable." % container[0]["confidence"]
        )

    return True, ("Peak critical-class confidence %.2f clears threshold %.2f."
                  % (max_critical, GUARDRAIL_THRESHOLD))


# --- LLM synthesis --------------------------------------------------------

SYSTEM_PROMPT = """You are a logistics exception adjudicator at a parcel sorting hub.

You receive three inputs: structured detections from a computer-vision model, the
immutable transit ledger record for the parcel, and an operator's question.

Rules you must follow:
- Reason only from the detections and ledger supplied. Never invent a detection,
  a confidence, a carrier, or a timestamp that is not in the input.
- If the vision findings contradict the ledger's origin status, say so explicitly
  and name the carrier that held custody between origin and here.
- If the inputs genuinely cannot answer the question, say that plainly.
- Answer in 2-4 sentences of plain operational English. No markdown, no lists.
"""


def _build_user_prompt(query: str, detections: List[dict], counts: Dict[str, int],
                       ledger_record: Optional[dict]) -> str:
    det_blob = json.dumps(detections, indent=2) if detections else "none"
    led_blob = json.dumps(ledger_record, indent=2) if ledger_record else "no record found for this package_id"
    return (
        "OPERATOR QUESTION:\n" + query + "\n\n"
        "VISION DETECTIONS (RT-DETR, absolute pixel boxes):\n" + det_blob + "\n\n"
        "CLASS COUNTS:\n" + json.dumps(counts) + "\n\n"
        "TRANSIT LEDGER RECORD:\n" + led_blob + "\n"
    )


def synthesize(query: str, detections: List[dict], counts: Dict[str, int],
               ledger_record: Optional[dict]) -> str:
    """
    Single direct call to the OpenAI SDK. No chains, no agents, no tool loop -
    we already gathered the context ourselves above.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _deterministic_fallback(detections, counts, ledger_record)

    try:
        from openai import OpenAI  # official SDK, imported directly

        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.1,
            max_tokens=300,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(query, detections, counts, ledger_record)},
            ],
        )
        return completion.choices[0].message.content.strip()
    except Exception as exc:  # network down, quota exhausted, bad key
        log.warning("LLM synthesis failed (%s); using deterministic fallback", exc)
        return _deterministic_fallback(detections, counts, ledger_record, error=str(exc))


def _deterministic_fallback(detections: List[dict], counts: Dict[str, int],
                            ledger_record: Optional[dict], error: str = None) -> str:
    """
    Template summary used when no API key is configured or the call fails.

    This keeps the endpoint reviewable without credentials. It is labelled as
    non-LLM output so nobody mistakes it for model reasoning.
    """
    from app.detector import CRITICAL_CLASSES

    defects = [d for d in detections if d["label"] in CRITICAL_CLASSES]
    prefix = "[deterministic fallback - LLM unavailable] "
    if error:
        prefix = "[deterministic fallback - LLM error: %s] " % error[:80]

    if not detections and ledger_record:
        # Ledger-only route: there were never any detections to describe, so
        # summarising "no objects observed" would answer a question nobody
        # asked. Report the custody record instead.
        hops = ledger_record.get("transit_history", [])
        body = ("Carrier of record is %s. Origin hub %s dispatched this parcel at %s "
                "with label status '%s' and seal status '%s', under manifest %s. "
                "The record shows %d custody scan(s)."
                % (ledger_record.get("carrier", "unknown"),
                   ledger_record.get("origin_hub", "unknown"),
                   ledger_record.get("dispatched_at", "unknown"),
                   ledger_record.get("origin_label_status", "unknown"),
                   ledger_record.get("origin_seal_status", "unknown"),
                   ledger_record.get("sku_manifest", "unknown"),
                   len(hops)))
    elif not defects:
        body = "No defect class detected. Observed objects: %s." % (counts or "none")
    else:
        listed = ", ".join("%s at %.2f" % (d["label"], d["confidence"]) for d in defects)
        body = "Detected %s." % listed
        if ledger_record:
            origin = ledger_record.get("origin_label_status", "unknown")
            carrier = ledger_record.get("carrier", "unknown carrier")
            body += (" Ledger records origin status '%s' under %s, so the damage "
                     "occurred in transit if origin was intact." % (origin, carrier))
    return prefix + body
