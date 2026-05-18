from app.cefr_data import CEFR_LEVELS

GRAMMAR_CUES = {
    "Question words": ["what", "where", "when", "who", "how", "which", "why"],
    "Adverbs of frequency": ["always", "usually", "often", "sometimes", "never", "every day", "every morning"],
    "Past simple": ["yesterday", "last", "ago", "did", "went", "had", "was", "were"],
    "Present continuous": ["am ", "is ", "are "],
    "Present perfect simple": ["have ever", "has ever", "have already", "has already", "have just", "has just"],
    "Prepositions of time and place": ["in ", "on ", "at ", "near", "next to", "behind", "under", "between"],
    "Comparatives and superlatives": ["more ", "most ", "better", "worse", "best", "than"],
    "Like + ing": ["like ", "love ", "enjoy "],
    "Future simple: will": ["will "],
    "Future: going to": ["going to"],
    "Past continuous": ["was ", "were "],
    "Present simple": ["every ", "usually ", "often "],
    "There is/There are": ["there is", "there are"],
    "Imperatives": ["please ", "ask ", "order ", "describe "],
    "Prepositions of place": ["near", "next to", "behind", "under", "between", "in front of"],
    "Modals: can, can't, do, doesn't": ["can ", "can't ", "do ", "doesn't "],
    "Simple adjectives": ["big", "small", "quiet", "noisy", "comfortable", "friendly"],
}


def normalize_cefr_level(level: str) -> str:
    normalized = level.strip().upper()

    if normalized not in CEFR_LEVELS:
        supported_levels = ", ".join(CEFR_LEVELS.keys())
        raise ValueError(f"Unsupported CEFR level '{level}'. Supported levels: {supported_levels}.")

    return normalized


def get_cefr_profile(level: str) -> dict:
    normalized_level = normalize_cefr_level(level)
    return CEFR_LEVELS[normalized_level]


def list_cefr_profiles() -> list[dict]:
    return [CEFR_LEVELS[level] for level in CEFR_LEVELS]


def get_cefr_focus_targets(level: str) -> list[str]:
    profile = get_cefr_profile(level)
    return [
        *profile["communication_targets"],
        *profile["vocabulary_targets"],
        *profile["grammar_targets"],
    ]


def get_cefr_targets_by_category(level: str) -> dict[str, list[str]]:
    profile = get_cefr_profile(level)
    return {
        "communication": profile["communication_targets"],
        "vocabulary": profile["vocabulary_targets"],
        "grammar": profile["grammar_targets"],
    }


def _normalize_focus_item_against_targets(item: str, targets: list[str]) -> str | None:
    normalized_input = item.strip().lower()

    if not normalized_input:
        return None

    for target in targets:
        normalized_target = target.lower()
        if normalized_input == normalized_target:
            return target
        if normalized_input in normalized_target or normalized_target in normalized_input:
            return target

    input_words = {word for word in normalized_input.replace("/", " ").replace("-", " ").split() if len(word) > 2}
    best_match = None
    best_score = 0.0

    for target in targets:
        target_words = {word for word in target.lower().replace("/", " ").replace("-", " ").split() if len(word) > 2}
        if not target_words:
            continue

        overlap = len(input_words & target_words) / len(target_words)
        if overlap > best_score:
            best_score = overlap
            best_match = target

    if best_match and best_score >= 0.34:
        return best_match

    return None


def normalize_cefr_focus_item(item: str, level: str) -> str:
    targets = get_cefr_focus_targets(level)
    match = _normalize_focus_item_against_targets(item, targets)
    if match:
        return match

    available_targets = ", ".join(targets)
    raise ValueError(
        f"Unsupported CEFR focus item '{item}' for level {level}. Allowed targets: {available_targets}."
    )


def choose_best_target_from_text(targets: list[str], text: str, fallback: str) -> str:
    normalized_text = text.lower()
    text_words = {word for word in normalized_text.replace("/", " ").replace("-", " ").split() if len(word) > 2}
    best_target = fallback
    best_score = -1.0

    for target in targets:
        target_words = {word for word in target.lower().replace("/", " ").replace("-", " ").split() if len(word) > 2}
        overlap_score = len(text_words & target_words)
        exact_score = 2 if target.lower() in normalized_text else 0
        score = overlap_score + exact_score

        if score > best_score:
            best_score = score
            best_target = target

    return best_target


def choose_best_grammar_target(targets: list[str], text: str) -> str:
    normalized_text = text.lower()
    best_target = targets[0]
    best_score = -1

    for target in targets:
        cues = GRAMMAR_CUES.get(target, [])
        score = sum(1 for cue in cues if cue in normalized_text)
        if score > best_score:
            best_score = score
            best_target = target

    if best_score <= 0:
        if "?" in normalized_text and "Question words" in targets:
            return "Question words"
        if "there is" in normalized_text or "there are" in normalized_text:
            there_is_target = _normalize_focus_item_against_targets("There is/There are", targets)
            if there_is_target:
                return there_is_target
        if "will " in normalized_text and "Future simple: will" in targets:
            return "Future simple: will"
        if "going to" in normalized_text and "Future: going to" in targets:
            return "Future: going to"
        if any(word in normalized_text for word in ["yesterday", "last ", "ago"]):
            past_target = _normalize_focus_item_against_targets("Past simple", targets)
            if past_target:
                return past_target
        if any(word in normalized_text for word in ["usually", "often", "always", "never"]):
            adverb_target = _normalize_focus_item_against_targets("Adverbs of frequency", targets)
            if adverb_target:
                return adverb_target

    return best_target


def format_cefr_profile_for_prompt(level: str) -> str:
    profile = get_cefr_profile(level)

    def render_lines(title: str, items: list[str]) -> str:
        lines = "\n".join(f"- {item}" for item in items)
        return f"{title}:\n{lines}"

    blocks = [
        f"CEFR level: {profile['level']} ({profile['label']})",
        f"Recommended study hours for the whole level: {profile['study_hours']}",
        render_lines("Can do", profile["can_do"]),
        render_lines("Communication targets", profile["communication_targets"]),
        render_lines("Vocabulary targets", profile["vocabulary_targets"]),
        render_lines("Grammar targets", profile["grammar_targets"]),
        f"Reference source: {profile['source_name']} - {profile['source_url']}",
    ]

    return "\n\n".join(blocks)
