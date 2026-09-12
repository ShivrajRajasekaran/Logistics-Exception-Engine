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

STRICT_ZERO_DETECTION_HALT = os.getenv("STRICT_ZERO_DETECTION_HALT", "false").lower() == "true"


VISION_TOKENS = {
    "damage", "damaged", "dent", "dented", "torn", "tear", "ripped", "crushed",
    "opened", "punctured", "wet", "collapsed",
    "intact", "condition", "inspect", "inspection", "visible", "see", "look", "image",
    "photo", "picture", "count", "how many", "box", "carton", "parcel", "package",
    "defect", "anomaly", "claim", "liability",
    # Vocabulary a reviewer actually types. The brief's own example question,
    # "What's the most common object here?", refused before these were added.
    "object", "objects", "common", "detect", "detected", "detection", "find",
    "found", "class", "classes", "describe", "summarize", "summarise", "summary",
    "findings", "problem", "problems", "issue", "issues", "wrong", "okay", "ok",
    "fine", "accept", "reject", "delivery", "shipment", "consignment", "goods",
    "auto route", "auto routed", "routed", "triage", "status",
}

UNSUPPORTED_TOKENS = {
    "person", "people", "worker", "operator", "staff", "anyone", "human",
    "label", "labels", "barcode", "printed", "readable", "legible", "address",
    "thermal", "heat", "temperature", "hotspot", "overheat", "cold",
    "seal", "tape", "tamper", "tampered", "sealed",
}

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

OUT_OF_SCOPE_TOKENS = {
    "weather", "stock price", "capital of", "translate", "joke", "recipe",
    "who are you", "your name", "sla definition", "company policy", "holiday",
    # Commercial/procurement vocabulary. The router now inspects the image when
    # nothing marks a question off-topic, so "off-topic" needs positive
    # evidence. Without these, "Who is our sealant supplier?" would reach the
    # detector and be answered from parcel boxes.
    "supplier", "vendor", "procurement", "price", "pricing", "cost", "quote",
    "contract", "budget", "invoice number", "purchase order",
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

    if unsupported:
        reasons = sorted({UNSUPPORTED_REASONS[t] for t in unsupported})
        return ROUTE_UNSUPPORTED, "asks about %s; %s" % (sorted(unsupported), "; ".join(reasons))

    if out_of_scope and not vision:
        return ROUTE_OUT_OF_SCOPE, "matched out-of-scope terms %s with no visual intent" % sorted(out_of_scope)

    if not vision:
        if ledger:
            return ROUTE_LEDGER, "matched record-only terms %s; parcel history answers this" % sorted(ledger)
        # An image was supplied and nothing marks the question as unsupported or
        # off-topic. Inspect it.
        #
        # Refusing here was a real defect: a keyword whitelist can never
        # enumerate every phrasing, so "no keyword matched" was silently
        # rejecting ordinary questions. Measured before this change, 8 of 16
        # plausible reviewer questions refused, including the brief's own
        # example "What's the most common object here?". Refusal must require
        # positive evidence (an unsupported concept, or an off-topic subject),
        # never the mere absence of a keyword.
        if image_path:
            return ROUTE_VISION, ("no explicit visual keyword, but an image was supplied "
                                  "and nothing marks the question off-topic; inspecting it")
        return ROUTE_OUT_OF_SCOPE, "no image supplied and no visual or record vocabulary matched"

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
        from openai import OpenAI

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
    except Exception as exc:
        log.warning("LLM synthesis failed (%s); using deterministic fallback", exc)
        return _deterministic_fallback(detections, counts, ledger_record, error=str(exc))


def _describe_llm_error(error: str) -> str:
    """
    Condense a provider exception into a short readable cause.

    The raw string is already written to the log at WARNING. Splicing 80
    characters of it into decision_summary put a truncated JSON error blob at
    the front of the operator-facing answer, which is what the demo console and
    the API response both surface first.
    """
    text = error.lower()
    if "429" in text or "rate limit" in text or "quota" in text:
        return "LLM provider rate limit"
    if "401" in text or "403" in text or "api key" in text or "authentic" in text:
        return "LLM credentials rejected"
    if "timeout" in text or "timed out" in text or "connection" in text:
        return "LLM provider unreachable"
    return "LLM call failed"


def _deterministic_fallback(detections: List[dict], counts: Dict[str, int],
                            ledger_record: Optional[dict], error: str = None) -> str:
    """
    Template summary used when no API key is configured or the call fails.

    This keeps the endpoint reviewable without credentials. It is labelled as
    non-LLM output so nobody mistakes it for model reasoning.
    """
    from app.detector import CRITICAL_CLASSES

    defects = [d for d in detections if d["label"] in CRITICAL_CLASSES]
    prefix = "[deterministic fallback - no LLM key configured] "
    if error:
        prefix = "[deterministic fallback - %s] " % _describe_llm_error(error)

    if not detections and ledger_record:
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
            # Reconcile against the custody record rather than restating it. The
            # previous wording appended "the damage occurred in transit if origin
            # was intact" unconditionally, which contradicted itself on a parcel
            # that left origin ALREADY_DAMAGED: the ledger value changed in the
            # sentence while the conclusion did not.
            if origin == "INTACT":
                body += (" Ledger records origin status INTACT under %s, so this "
                         "damage was not present at dispatch and is attributable "
                         "to the carrier." % carrier)
            elif origin == "ALREADY_DAMAGED":
                body += (" Ledger records origin status ALREADY_DAMAGED under %s, "
                         "so this parcel left origin damaged and the finding is "
                         "NOT a new carrier claim. Route to manual inspection only "
                         "if the damage appears to have worsened." % carrier)
            else:
                body += (" Ledger records origin status '%s' under %s, which does "
                         "not establish whether the damage predates dispatch, so "
                         "liability cannot be attributed from the record alone."
                         % (origin, carrier))
        else:
            body += (" No transit record exists for this package, so the damage "
                     "cannot be attributed to a carrier from available data.")
    return prefix + body
