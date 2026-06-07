"""Shared semantic guards for fine-grained why/action-intent tasks."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable


POST_ACTION_SENSITIVE_PATTERNS = (
    r"\bpick(?:ed)?\s+up\b",
    r"\btake(?:n|s)?\b",
    r"\btook\b",
    r"\bgrab(?:bed)?\b",
    r"\blift(?:ed)?\b",
    r"\bhold(?:ing)?\b",
    r"\bcarry\b",
    r"\bmove(?:d)?\b",
    r"\bshift(?:ed)?\b",
    r"\bremove(?:d)?\b",
    r"\bclear(?:ed)?\b",
    r"\bput\b",
    r"\bplace(?:d)?\b",
    r"\breturn(?:ed)?\b",
    r"\bopen(?:ed)?\b",
    r"\bclose(?:d)?\b",
    r"\bturn(?:ed)?\b",
    r"\bpour(?:ed)?\b",
    r"\btip(?:ped)?\b",
    r"\bempty\b",
    r"\bwash(?:ed)?\b",
    r"\bclean(?:ed)?\b",
    r"\bwipe(?:d)?\b",
    r"\bdry\b",
    r"\bcut\b",
    r"\bpeel(?:ed)?\b",
    r"\bmix(?:ed)?\b",
    r"\bstir(?:red)?\b",
    r"\badd(?:ed)?\b",
    r"\bscoop(?:ed)?\b",
    r"\bscrape(?:d)?\b",
    r"\bflip(?:ped)?\b",
    r"\bshake(?:n|s|d)?\b",
    r"\btilt(?:ed)?\b",
    r"\bthrow\b",
    r"\bdiscard\b",
    r"\btap(?:ped)?\b",
    r"\bpress(?:ed)?\b",
    r"\bpush(?:ed)?\b",
)


CHOICE_CATEGORY_PATTERNS: dict[str, tuple[str, ...]] = {
    "generic_relocation": (
        r"\bto move\b",
        r"\bmove\.$",
        r"\bmove it\b",
        r"\bmove the\b",
        r"\brelocate\b",
        r"\bshift it\b",
        r"\bshift the\b",
        r"\bset aside\b",
        r"\btemporarily relocate\b",
    ),
    "access_retrieve": (
        r"\baccess\b",
        r"\bbehind\b",
        r"\bretrieve\b",
        r"\bget\s+to\b",
        r"\bget\s+at\b",
        r"\bget\s+out\b",
        r"\bget\s+back\b",
        r"\btake\s+out\b",
        r"\bpick\s+up\b",
        r"\bmissing\b",
        r"\breach\b",
        r"\bclear\s+the\s+way\b",
    ),
    "space_clear": (
        r"\bmake\s+space\b",
        r"\bcreate\s+space\b",
        r"\bfree\s+up\s+space\b",
        r"\bclear\s+space\b",
        r"\bmake\s+room\b",
        r"\bcreate\s+room\b",
        r"\bout\s+of\s+the\s+way\b",
        r"\bclear\s+the\s+counter\b",
    ),
    "final_place_return": (
        r"\bput\s+back\b",
        r"\bput\b",
        r"\bplace\b",
        r"\bright\s+place\b",
        r"\bproper\s+place\b",
        r"\breturn\b",
        r"\bstore\b",
        r"\baway\b",
        r"\bin\s+place\b",
        r"\bslot\b",
        r"\binsert\b",
    ),
    "measure_weigh": (
        r"\bweigh\b",
        r"\bmeasure\b",
        r"\bmeasurement(?:s)?\b",
        r"\badjust\b",
        r"\bscale\b",
        r"\bgrams?\b",
        r"\breading\b",
    ),
    "transfer_contents": (
        r"\bempty\b",
        r"\bpour\b",
        r"\bdrain\b",
        r"\btip\b",
        r"\bfill\b",
        r"\btransfer\b",
        r"\binto\b",
        r"\bdrop\b",
        r"\bfall\s+off\b",
        r"\bfall\s+back\b",
        r"\bexcess\b",
        r"\bshake\s+off\b",
        r"\brelease\b",
    ),
    "serve_consume": (
        r"\bserve\b",
        r"\bplate\b",
        r"\beat\b",
        r"\btaste\b",
        r"\bdrink\b",
        r"\bportion\b",
    ),
    "clean_dry": (
        r"\bwash\b",
        r"\bclean\b",
        r"\bwipe\b",
        r"\bdry\b",
        r"\brinse\b",
        r"\btowel\b",
        r"\bcloth\b",
    ),
    "inspect_check": (
        r"\bcheck\b",
        r"\binspect\b",
        r"\blook\b",
        r"\bread\b",
        r"\bscan\b",
        r"\bsee\b",
        r"\bfind\b",
    ),
    "open_close": (
        r"\bopen\b",
        r"\bclose\b",
        r"\bturn\s+on\b",
        r"\bturn\s+off\b",
        r"\bturn(?:ed)?\s+.*?\s+on\b",
        r"\bturn(?:ed)?\s+.*?\s+off\b",
        r"\bswitch\s+on\b",
        r"\bswitch\s+off\b",
        r"\bswitch\b",
        r"\buncap\b",
        r"\bcap\b",
        r"\blid\b",
        r"\bunscrew\b",
        r"\bopen\s+up\b",
    ),
    "hand_free_enablement": (
        r"\bfree\s+hand\b",
        r"\bother\s+hand\b",
        r"\bleft\s+hand\b",
        r"\bright\s+hand\b",
        r"\bwith\s+left\s+hand\b",
        r"\bwith\s+right\s+hand\b",
        r"\bone\s+hand\b",
        r"\btwo\s+hands?\b",
    ),
    "food_prep": (
        r"\bmix\b",
        r"\bstir\b",
        r"\bcut\b",
        r"\bchop\b",
        r"\bpeel\b",
        r"\bcook\b",
        r"\badd\b",
        r"\bseason\b",
        r"\bspread\b",
        r"\bwrap\b",
    ),
    "discard": (
        r"\bthrow\b",
        r"\bdiscard\b",
        r"\btrash\b",
        r"\bbin\b",
        r"\bgarbage\b",
        r"\bdispose\b",
    ),
    "safety_avoid": (
        r"\bavoid\b",
        r"\bhot\b",
        r"\bburn\b",
        r"\bsafe\b",
        r"\bsafety\b",
        r"\bspill\b",
        r"\bmess\b",
        r"\bmessy\b",
        r"\bdirty\b",
    ),
}

PAIRWISE_OUTCOME_CATEGORIES = frozenset(
    {"generic_relocation", "access_retrieve", "space_clear", "final_place_return", "safety_avoid"}
)
PRECONDITION_CATEGORIES = frozenset({"clean_dry", "safety_avoid"})
FUTURE_USE_CATEGORIES = frozenset(
    {
        "measure_weigh",
        "transfer_contents",
        "serve_consume",
        "clean_dry",
        "inspect_check",
        "open_close",
        "food_prep",
        "discard",
        "final_place_return",
        "access_retrieve",
        "hand_free_enablement",
    }
)


def normalize_action_intent_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _extract_question_action_text(question: str) -> str:
    lowered = normalize_action_intent_text(question)
    match = re.search(r"<([^>]+)>", lowered)
    if match:
        return normalize_action_intent_text(match.group(1))
    return lowered


def _question_action_has_any(question: str, tokens: Iterable[str]) -> bool:
    action_text = _extract_question_action_text(question)
    return any(token in action_text for token in tokens)


def _question_mentions_towel_like_object(question: str) -> bool:
    return _question_action_has_any(
        question,
        ("towel", "cloth", "napkin", "paper towel", "tea towel", "dish cloth", "hand towel"),
    )


def _question_requires_transport_vs_use_evidence(question: str) -> bool:
    return _question_mentions_towel_like_object(question) and _question_action_has_any(
        question,
        ("pick up", "grab", "lift", "take", "move", "shift"),
    )


def _question_requires_residue_release_evidence(question: str) -> bool:
    return _question_mentions_towel_like_object(question) and _question_action_has_any(
        question,
        ("flip", "turn", "shake", "tip"),
    )


def _question_requires_state_change_evidence(question: str) -> bool:
    return _question_action_has_any(question, ("tap", "press", "push")) and _question_action_has_any(
        question,
        ("scale", "button", "switch", "knob"),
    )


def question_is_post_action_sensitive(question: str) -> bool:
    text = normalize_action_intent_text(question)
    return any(re.search(pattern, text) for pattern in POST_ACTION_SENSITIVE_PATTERNS)


def choice_categories(choice: str) -> set[str]:
    text = normalize_action_intent_text(choice)
    categories: set[str] = set()
    for category, patterns in CHOICE_CATEGORY_PATTERNS.items():
        if any(re.search(pattern, text) for pattern in patterns):
            categories.add(category)
    return categories


def selected_choice_categories(choices: list[str], indices: Iterable[int] | None = None) -> dict[int, set[str]]:
    if indices is None:
        normalized_indices = range(len(choices))
    else:
        normalized_indices = []
        for raw_index in indices:
            try:
                index = int(raw_index)
            except Exception:  # noqa: BLE001
                continue
            if 0 <= index < len(choices):
                normalized_indices.append(index)
    return {index: choice_categories(str(choices[index])) for index in normalized_indices}


def _choice_is_generic_hidden_access(choice: str) -> bool:
    text = normalize_action_intent_text(choice)
    if "behind" not in text and "access" not in text:
        return False
    if any(
        token in text
        for token in (
            "take the",
            "retrieve the",
            "pick up the",
            "look for the",
            "find the",
            "put the",
            "place the",
            "right place",
            "proper place",
            "freed slot",
        )
    ):
        return False
    return any(
        token in text
        for token in (
            "access what's behind",
            "access what is behind",
            "access behind",
            "access the slot behind",
            "access the area behind",
            "see what is behind",
            "see what's behind",
            "look what's behind",
            "look behind",
            "what is behind",
            "what's behind",
            "behind the",
        )
    )


def _choice_is_exact_reveal_target_use(choice: str) -> bool:
    text = normalize_action_intent_text(choice)
    if any(
        token in text
        for token in (
            "take the",
            "retrieve the",
            "pick up the",
            "look for the",
            "find the",
            "grab the",
            "take the hidden",
            "missing",
            "put the",
            "place the",
            "right place",
            "proper place",
            "freed slot",
            "slot",
        )
    ):
        return True
    return False


def action_intent_conflict_profile(
    *,
    question: str,
    choices: list[str],
    indices: Iterable[int] | None = None,
) -> dict[str, object]:
    by_index = selected_choice_categories(choices, indices)
    category_counts = Counter(category for categories in by_index.values() for category in categories)
    active_categories = {category for category, count in category_counts.items() if count > 0}
    pairwise_categories = active_categories & PAIRWISE_OUTCOME_CATEGORIES
    future_categories = active_categories & FUTURE_USE_CATEGORIES
    has_access_space_conflict = "access_retrieve" in active_categories and "space_clear" in active_categories
    has_space_place_conflict = "space_clear" in active_categories and "final_place_return" in active_categories
    has_access_place_conflict = "access_retrieve" in active_categories and "final_place_return" in active_categories
    selected_texts = {index: normalize_action_intent_text(str(choices[index])) for index in by_index}
    has_hidden_access_exact_use_conflict = any(
        _choice_is_generic_hidden_access(text)
        for text in selected_texts.values()
    ) and any(
        _choice_is_exact_reveal_target_use(text)
        for text in selected_texts.values()
    )
    return {
        "post_action_sensitive": question_is_post_action_sensitive(question),
        "categories_by_index": by_index,
        "category_counts": dict(category_counts),
        "active_categories": active_categories,
        "future_categories": future_categories,
        "pairwise_categories": pairwise_categories,
        "has_hidden_access_exact_use_conflict": has_hidden_access_exact_use_conflict,
        "has_pairwise_outcome_conflict": (
            has_access_space_conflict
            or has_space_place_conflict
            or has_access_place_conflict
            or has_hidden_access_exact_use_conflict
            or len(pairwise_categories) >= 2
        ),
        "has_future_use_conflict": len(future_categories) >= 2,
    }


def action_intent_followup_decision(
    *,
    question: str,
    choices: list[str],
    indices: Iterable[int] | None = None,
    confidence: float = 1.0,
    reason_text: str = "",
) -> tuple[bool, str, float, str]:
    """Return (needs_followup, reason, window_s, resolver)."""

    profile = action_intent_conflict_profile(question=question, choices=choices, indices=indices)
    full_profile = action_intent_conflict_profile(question=question, choices=choices, indices=None)
    if not bool(profile["post_action_sensitive"]):
        return False, "", 4.0, ""
    active_categories = set(profile["active_categories"])
    future_categories = set(profile["future_categories"])
    full_active_categories = set(full_profile["active_categories"])
    full_category_counts = dict(full_profile["category_counts"])
    non_pairwise_future_categories = future_categories - PAIRWISE_OUTCOME_CATEGORIES
    candidate_count = len(profile["categories_by_index"])
    has_pairwise_outcome_conflict = bool(profile["has_pairwise_outcome_conflict"])
    has_future_use_conflict = bool(profile["has_future_use_conflict"])
    has_hidden_access_exact_use_conflict = bool(profile["has_hidden_access_exact_use_conflict"])
    if _question_requires_state_change_evidence(question) and {
        "open_close",
        "measure_weigh",
    }.issubset(active_categories):
        return True, "state_change_disambiguation_needed", 6.0, "pairwise"
    if _question_requires_residue_release_evidence(question) and {
        "clean_dry",
        "transfer_contents",
    }.issubset(active_categories):
        return True, "residue_release_vs_cleanup_needed", 6.0, "pairwise"
    if _question_requires_transport_vs_use_evidence(question) and {
        "clean_dry",
        "generic_relocation",
    }.issubset(active_categories):
        return True, "transport_vs_cleanup_needed", 6.0, "pairwise"
    if _question_requires_transport_vs_use_evidence(question) and "clean_dry" in full_active_categories:
        clean_dry_candidate_count = int(full_category_counts.get("clean_dry", 0))
        has_full_relocation_candidate = "generic_relocation" in full_active_categories or "final_place_return" in full_active_categories
        missing_relocation_in_current_candidates = "generic_relocation" not in active_categories and "final_place_return" not in active_categories
        if clean_dry_candidate_count >= 2 and has_full_relocation_candidate and (
            candidate_count > 2 or missing_relocation_in_current_candidates
        ):
            return True, "transport_vs_cleanup_future_use_needed", 8.0, "future_use"
    if has_pairwise_outcome_conflict and candidate_count <= 2:
        return True, "outcome_dependent_pairwise_needed", 4.0, "pairwise"
    if has_hidden_access_exact_use_conflict:
        return True, "hidden_access_exact_use_pairwise_needed", 4.0, "pairwise"
    if has_future_use_conflict and (non_pairwise_future_categories or not has_pairwise_outcome_conflict):
        return True, "future_use_evidence_needed", 8.0, "future_use"
    if has_pairwise_outcome_conflict:
        return True, "outcome_dependent_pairwise_needed", 4.0, "pairwise"
    if len(active_categories) >= 2 and confidence < 0.86:
        return True, "low_confidence_multi_intent_conflict", 6.0, "future_use"
    uncertainty = normalize_action_intent_text(reason_text)
    if any(term in uncertainty for term in ("unclear", "uncertain", "ambiguous", "cannot tell", "can't tell", "not visible")):
        return True, "reason_explicitly_uncertain", 6.0, "future_use" if future_categories else "pairwise"
    return False, "", 4.0, ""


def action_intent_needs_future_use_resolution(
    *,
    question: str,
    choices: list[str],
    indices: Iterable[int] | None = None,
) -> bool:
    needs, _, _, resolver = action_intent_followup_decision(
        question=question,
        choices=choices,
        indices=indices,
    )
    return bool(needs and resolver == "future_use")


def action_intent_needs_pairwise_resolution(
    *,
    question: str,
    choices: list[str],
    indices: Iterable[int] | None = None,
) -> bool:
    needs, _, _, resolver = action_intent_followup_decision(
        question=question,
        choices=choices,
        indices=indices,
    )
    return bool(needs and resolver == "pairwise")


def action_intent_requires_strict_visual_disambiguation(
    *,
    question: str,
    choices: list[str],
    indices: Iterable[int] | None = None,
) -> bool:
    profile = action_intent_conflict_profile(question=question, choices=choices, indices=indices)
    active_categories = set(profile["active_categories"])
    if _question_requires_state_change_evidence(question) and {
        "open_close",
        "measure_weigh",
    }.issubset(active_categories):
        return True
    if _question_requires_residue_release_evidence(question) and {
        "clean_dry",
        "transfer_contents",
    }.issubset(active_categories):
        return True
    if _question_requires_transport_vs_use_evidence(question) and "clean_dry" in active_categories:
        if {"generic_relocation", "final_place_return"} & active_categories:
            return True
    return False


def action_intent_needs_precondition_context(
    *,
    question: str,
    choices: list[str],
    indices: Iterable[int] | None = None,
) -> bool:
    profile = action_intent_conflict_profile(question=question, choices=choices, indices=indices)
    if not bool(profile["post_action_sensitive"]):
        return False
    active_categories = set(profile["active_categories"])
    if "clean_dry" in active_categories:
        return True
    if "safety_avoid" in active_categories:
        return True
    return False
