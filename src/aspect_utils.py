import re

# Canonical aspect labels used across the app
ASPECT_LABELS = [
    'doctor',
    'wait',
    'facility',
    'treatment',
    'staff',
]

# Keyword dictionary to bootstrap weak labels and for fallbacks
ASPECT_KEYWORDS = {
    'doctor': ['doctor', 'physician', 'specialist', 'consultation', 'diagnosis', 'treatment plan'],
    'wait': ['wait', 'waiting', 'delay', 'appointment', 'schedule', 'time'],
    'facility': ['facility', 'hospital', 'clinic', 'room', 'clean', 'hygiene', 'equipment'],
    'treatment': ['treatment', 'medication', 'therapy', 'procedure', 'recovery', 'healing'],
    'staff': ['staff', 'nurse', 'receptionist', 'assistant', 'personnel', 'team']
}


def heuristic_detect_aspect(text: str) -> str:
    """Heuristic aspect detection using keyword counts.
    Returns one of ASPECT_LABELS or 'overall'.
    """
    lowered = (text or '').lower()
    best_aspect = 'overall'
    best_score = 0
    for aspect, keywords in ASPECT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in lowered)
        if score > best_score:
            best_score = score
            best_aspect = aspect
    return best_aspect


def extract_aspect_relevant_text(text: str, aspect: str) -> str:
    if aspect == 'overall':
        return text
    keywords = [k.lower() for k in ASPECT_KEYWORDS.get(aspect, [])]
    separators = re.compile(r"(?<=[.!?])\s+|\s+but\s+|\s+however\s+|\s+although\s+|,\s+", re.IGNORECASE)
    segments = [seg.strip() for seg in separators.split(text) if seg and seg.strip()]
    relevant = [seg for seg in segments if any(kw in seg.lower() for kw in keywords)]
    return ". ".join(relevant) if relevant else text

