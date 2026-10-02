"""Segment English into natural reading chunks for language learners.

Helps learners understand long sentences by marking clause / phrase
boundaries (main, relative, dependent, prepositional, etc.). No NLP
packages — heuristic splits aimed at readable break points.
"""

from __future__ import annotations

import re
from typing import TypedDict


class GrammarSegment(TypedDict):
    text: str
    label: str


# Clause / phrase labels (also used as color keys in the UI).
MAIN_CLAUSE = "main_clause"
DEPENDENT_CLAUSE = "dependent_clause"
RELATIVE_CLAUSE = "relative_clause"
PREPOSITIONAL_PHRASE = "prepositional_phrase"
NOUN_PHRASE = "noun_phrase"
VERB_PHRASE = "verb_phrase"
PARTICIPIAL_PHRASE = "participial_phrase"
INFINITIVE_PHRASE = "infinitive_phrase"

# Short learner-facing names shown above each chunk.
LEARNER_LABELS = {
    MAIN_CLAUSE: "main",
    DEPENDENT_CLAUSE: "dependent",
    RELATIVE_CLAUSE: "relative",
    PREPOSITIONAL_PHRASE: "prep phrase",
    NOUN_PHRASE: "noun phrase",
    VERB_PHRASE: "verb phrase",
    PARTICIPIAL_PHRASE: "participial",
    INFINITIVE_PHRASE: "infinitive",
}

_RELATIVE = frozenset({"which", "who", "whom", "whose", "that", "where"})
_DEPENDENT = frozenset(
    {
        "because",
        "although",
        "though",
        "while",
        "whereas",
        "if",
        "unless",
        "since",
        "after",
        "before",
        "when",
        "whenever",
        "wherever",
        "until",
        "once",
        "as",
        "so",
        "than",
        "whether",
        "even",
    }
)
_PREP = frozenset(
    {
        "of",
        "in",
        "on",
        "at",
        "for",
        "with",
        "from",
        "by",
        "about",
        "into",
        "onto",
        "upon",
        "over",
        "under",
        "through",
        "during",
        "without",
        "within",
        "among",
        "between",
        "against",
        "across",
        "around",
        "along",
        "toward",
        "towards",
        "via",
        "per",
        "like",
        "unlike",
        "despite",
        "beside",
        "besides",
        "near",
        "above",
        "below",
        "beyond",
        "inside",
        "outside",
        "throughout",
    }
)

_AUX = frozenset(
    {
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "can",
        "could",
        "may",
        "might",
        "must",
    }
)

_INFINITIVE_VERBS = frozenset(
    {
        "be",
        "have",
        "do",
        "make",
        "get",
        "go",
        "see",
        "know",
        "take",
        "come",
        "think",
        "look",
        "want",
        "give",
        "use",
        "find",
        "tell",
        "ask",
        "work",
        "seem",
        "feel",
        "try",
        "leave",
        "call",
        "keep",
        "let",
        "begin",
        "help",
        "show",
        "hear",
        "play",
        "run",
        "move",
        "live",
        "believe",
        "hold",
        "bring",
        "happen",
        "write",
        "provide",
        "sit",
        "stand",
        "lose",
        "pay",
        "meet",
        "include",
        "continue",
        "set",
        "learn",
        "change",
        "lead",
        "understand",
        "watch",
        "follow",
        "stop",
        "create",
        "speak",
        "read",
        "allow",
        "add",
        "spend",
        "grow",
        "open",
        "walk",
        "win",
        "offer",
        "remember",
        "love",
        "consider",
        "appear",
        "buy",
        "wait",
        "serve",
        "die",
        "send",
        "expect",
        "build",
        "stay",
        "fall",
        "cut",
        "reach",
        "kill",
        "remain",
        "suggest",
        "raise",
        "pass",
        "sell",
        "require",
        "report",
        "decide",
        "pull",
        "explain",
        "hope",
        "develop",
        "carry",
        "break",
        "receive",
        "agree",
        "support",
        "hit",
        "produce",
        "eat",
        "cover",
        "catch",
        "draw",
        "choose",
        "study",
        "improve",
        "focus",
        "explore",
        "discuss",
        "describe",
        "avoid",
        "achieve",
        "ensure",
        "maintain",
        "reduce",
        "increase",
        "start",
        "finish",
        "become",
        "enable",
        "succeed",
        "turn",
    }
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\s\w]", re.UNICODE)

# Tiny 1–2 word scraps merge with a neighbor; real phrases keep their · breaks.
MIN_CHUNK_WORDS = 2
_BREAK_LABELS = {"sentence_break", "paragraph_break"}
_COORD = frozenset({"and", "but", "or", "nor", "yet"})


def learner_label(label: str) -> str:
    """Short display name for English learners."""
    return LEARNER_LABELS.get(label, label.replace("_", " "))


def _tokenize(sentence: str) -> list[str]:
    return _TOKEN_RE.findall(sentence)


def _norm(tok: str) -> str:
    return tok.lower()


def _is_word(tok: str) -> bool:
    return bool(re.match(r"[A-Za-z]", tok))


_ATTACH_LEFT = frozenset({",", ".", "!", "?", ";", ":", "%", "'", "’", ")", "]", "}"})
_ATTACH_RIGHT = frozenset({"(", "[", "{", "$", "#", "@", "“", '"'})


def _join_tokens(tokens: list[str]) -> str:
    """Rejoin tokens with sensible spacing around punctuation."""
    if not tokens:
        return ""
    out: list[str] = []
    for i, tok in enumerate(tokens):
        if i == 0:
            out.append(tok)
            continue
        prev = out[-1]
        if tok in _ATTACH_LEFT:
            out.append(tok)
        elif prev in _ATTACH_RIGHT:
            out.append(tok)
        else:
            out.append(" " + tok)
    return "".join(out).strip()


def _glue_texts(left: str, right: str) -> str:
    """Join segment texts without inserting a space before trailing punctuation."""
    left = (left or "").rstrip()
    right = (right or "").lstrip()
    if not left:
        return right
    if not right:
        return left
    if right[0] in _ATTACH_LEFT:
        return left + right
    if left[-1] in _ATTACH_RIGHT:
        return left + right
    return f"{left} {right}"


def _looks_finite_verb(tok: str) -> bool:
    low = _norm(tok)
    if low in _AUX:
        return True
    if low in _INFINITIVE_VERBS:
        return True
    if low.endswith("ing"):
        return True
    if low.endswith("ed") and low not in {"red", "bed", "need", "feed", "used"}:
        return True
    return False


def _is_verb_before_prep(tok: str) -> bool:
    """Conservative verb check for PP splits (avoid noun plurals like components)."""
    low = _norm(tok)
    if low in _AUX or low in _INFINITIVE_VERBS:
        return True
    if low.endswith("ing"):
        return True
    if low.endswith("ed") and low not in {"red", "bed", "need", "feed", "used"}:
        return True
    return False


def _find_clause_split(
    tokens: list[str], starters: frozenset[str], *, skip_first: bool = True
) -> int | None:
    """Index of a clause-starter word (not the first content word if skip_first)."""
    seen_word = False
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if _is_word(t):
            if seen_word or not skip_first:
                low = _norm(t)
                if low in starters:
                    if low == "even" and i + 1 < len(tokens):
                        nxt = _norm(tokens[i + 1])
                        if nxt in {"if", "though", "when", "after", "before"}:
                            return i
                        i += 1
                        continue
                    if low == "as" and i + 1 < len(tokens) and _norm(tokens[i + 1]) == "if":
                        return i
                    if low == "so" and i + 1 < len(tokens) and _norm(tokens[i + 1]) == "that":
                        return i
                    # Avoid splitting on determiner "that" (that book / that interesting…).
                    if low == "that" and i + 1 < len(tokens):
                        nxt = _norm(tokens[i + 1])
                        if nxt in {
                            "a",
                            "an",
                            "the",
                            "this",
                            "these",
                            "those",
                            "my",
                            "your",
                            "his",
                            "her",
                            "our",
                            "their",
                        }:
                            i += 1
                            continue
                        if nxt not in {
                            "i",
                            "you",
                            "he",
                            "she",
                            "it",
                            "we",
                            "they",
                            "is",
                            "are",
                            "was",
                            "were",
                            "has",
                            "have",
                            "had",
                            "will",
                            "would",
                            "can",
                            "could",
                            "may",
                            "might",
                            "must",
                            "does",
                            "do",
                            "did",
                        } and not _looks_finite_verb(tokens[i + 1]):
                            # Likely determiner before a noun phrase.
                            i += 1
                            continue
                    return i
            seen_word = True
        i += 1
    return None


def _find_relative_split(tokens: list[str]) -> int | None:
    return _find_clause_split(tokens, _RELATIVE, skip_first=True)


def _find_dependent_split(tokens: list[str]) -> int | None:
    for i, tok in enumerate(tokens):
        if i == 0:
            continue
        if not _is_word(tok):
            continue
        low = _norm(tok)
        if low == "even" and i + 1 < len(tokens) and _norm(tokens[i + 1]) in {
            "if",
            "though",
            "when",
        }:
            return i
        if low == "so" and i + 1 < len(tokens) and _norm(tokens[i + 1]) == "that":
            return i
    return _find_clause_split(tokens, _DEPENDENT - {"when"}, skip_first=True)


def _relative_clause_end(tokens: list[str], rel_start: int) -> int:
    """Where a relative clause likely ends so the outer clause can resume.

    Learner-friendly: "The book that she recommended last week | turned out…"
    """
    finite = 0
    i = rel_start + 1
    while i < len(tokens):
        if not _is_word(tokens[i]):
            i += 1
            continue
        low = _norm(tokens[i])
        if finite == 0 and low in {"i", "you", "he", "she", "it", "we", "they"}:
            i += 1
            continue
        if _looks_finite_verb(tokens[i]):
            finite += 1
            if finite >= 2:
                return i
        i += 1
    return len(tokens)


def _find_participial_split(tokens: list[str]) -> int | None:
    """Return index where a mid-sentence participial phrase begins, if any."""
    for i in range(1, len(tokens) - 1):
        if tokens[i] != ",":
            continue
        nxt = tokens[i + 1]
        if not _is_word(nxt):
            continue
        low = _norm(nxt)
        if low.endswith("ing") or (
            low.endswith("ed") and low not in {"red", "bed", "need", "feed"}
        ):
            return i + 1
    return None


def _find_sentence_initial_phrase_end(tokens: list[str]) -> tuple[int, str] | None:
    """Sentence-initial dependent / infinitive / participial before a comma."""
    if not tokens or not _is_word(tokens[0]):
        return None
    low0 = _norm(tokens[0])
    head_label = None

    if low0 in _DEPENDENT or (
        low0 == "even"
        and len(tokens) > 1
        and _norm(tokens[1]) in {"if", "though", "when"}
    ):
        head_label = DEPENDENT_CLAUSE
    elif low0 == "to" and len(tokens) > 2 and _is_word(tokens[1]):
        nlow = _norm(tokens[1])
        if nlow not in {
            "the",
            "a",
            "an",
            "this",
            "that",
            "my",
            "your",
            "his",
            "her",
            "our",
            "their",
        }:
            head_label = INFINITIVE_PHRASE
    elif low0.endswith("ing") or low0.endswith("ed"):
        head_label = PARTICIPIAL_PHRASE

    if head_label is None:
        return None
    try:
        comma = tokens.index(",", 1)
    except ValueError:
        return None
    if not (1 <= comma <= 16):
        return None
    j = comma + 1
    while j < len(tokens) and not _is_word(tokens[j]):
        j += 1
    if j >= len(tokens):
        return None
    return j, head_label


def _find_infinitive_split(tokens: list[str]) -> int | None:
    """Mid-sentence infinitive that forms a natural reading chunk."""
    for i in range(1, len(tokens) - 1):
        if _norm(tokens[i]) != "to":
            continue
        nxt = tokens[i + 1]
        if not _is_word(nxt):
            continue
        nlow = _norm(nxt)
        if nlow in {
            "the",
            "a",
            "an",
            "this",
            "that",
            "these",
            "those",
            "my",
            "your",
            "his",
            "her",
            "our",
            "their",
        }:
            continue
        if nlow in _INFINITIVE_VERBS or nlow.endswith(("ate", "ize", "ise", "ify")):
            words = 1
            end = i + 1
            while end + 1 < len(tokens) and _is_word(tokens[end + 1]):
                if _norm(tokens[end + 1]) in _RELATIVE | (_DEPENDENT - {"as"}):
                    break
                end += 1
                words += 1
                if words >= 8:
                    break
            if words >= 2:
                return i
    return None


def _find_comma_pause_splits(tokens: list[str]) -> list[int]:
    """Break after commas that separate list items or reading pauses.

    Example: \"…architecture, transaction processing, database design…\"
    → cut at the token after each comma.
    """
    splits: list[int] = []
    words_before = 0
    for i, tok in enumerate(tokens):
        if _is_word(tok):
            words_before += 1
            continue
        if tok != ",":
            continue
        # Need a real head before the comma.
        if words_before < 3:
            continue
        j = i + 1
        while j < len(tokens) and not _is_word(tokens[j]):
            j += 1
        if j >= len(tokens):
            continue
        # Skip \"and/or\" immediately after comma for now — coordinating handler
        # will pick those up; still allow cut at the item start.
        words_after = 0
        k = j
        while k < len(tokens) and tokens[k] != ",":
            if _is_word(tokens[k]):
                words_after += 1
            k += 1
            if words_after >= 8:
                break
        if words_after >= 2:
            splits.append(j)
    return splits


def _find_coord_splits(tokens: list[str]) -> list[int]:
    """Break before coordinating and/but/or when both sides are substantial."""
    splits: list[int] = []
    word_idx = 0
    for i, tok in enumerate(tokens):
        if not _is_word(tok):
            continue
        word_idx += 1
        low = _norm(tok)
        if low not in _COORD:
            continue
        # Skip very early coordinators (\"and\" in subject) and tiny left sides.
        if word_idx < 4:
            continue
        # Left side should already have a verb-ish feel or several words.
        left_words = [t for t in tokens[:i] if _is_word(t)]
        right_words = [t for t in tokens[i + 1 :] if _is_word(t)]
        if len(left_words) < 4 or len(right_words) < 2:
            continue
        # Avoid splitting \"X and Y\" inside a short NP: require the left side
        # to contain an auxiliary/main verb or a prior comma pause.
        left_has_verb = any(_looks_finite_verb(w) for w in left_words)
        left_has_comma = "," in tokens[:i]
        if not left_has_verb and not left_has_comma:
            continue
        splits.append(i)
    return splits


def _word_before(tokens: list[str], idx: int) -> int | None:
    j = idx - 1
    while j >= 0:
        if _is_word(tokens[j]):
            return j
        j -= 1
    return None


def _allow_prepositional_split(tokens: list[str], prep_idx: int) -> bool:
    """Only break before adverbial PPs (after a verb), not noun-attached PPs.

    Learners must read \"different parts of the system\" as one unit — never
    \"parts · of the system\".
    """
    prep = _norm(tokens[prep_idx])
    prev_i = _word_before(tokens, prep_idx)
    if prev_i is None:
        return False

    prev = _norm(tokens[prev_i])

    # Multi-word prepositions: never split on the second word.
    if prev in _DEPENDENT | {"out", "because", "instead", "regardless", "according"}:
        return False

    # \"of\" almost always completes the preceding noun phrase.
    if prep == "of":
        return False

    # PPs that follow a verb are usually adverbial reading pauses.
    if _is_verb_before_prep(tokens[prev_i]):
        return True

    # \"for monitoring\", \"with care\", \"in mind\" after a noun → NP modifier.
    if prep in {"for", "with", "in", "on", "at", "from", "by", "about", "into"}:
        return False

    return False


def _find_prepositional_splits(tokens: list[str]) -> list[int]:
    """Indices where an adverbial prepositional phrase begins."""
    splits: list[int] = []
    seen_word = False
    words_seen = 0
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if _is_word(t):
            words_seen += 1
            low = _norm(t)
            if seen_word and words_seen >= 3 and low in _PREP:
                if low == "to" and i + 1 < len(tokens):
                    nxt = _norm(tokens[i + 1])
                    if nxt in _INFINITIVE_VERBS or (
                        _is_word(tokens[i + 1])
                        and nxt
                        not in {
                            "the",
                            "a",
                            "an",
                            "this",
                            "that",
                            "my",
                            "your",
                            "his",
                            "her",
                            "our",
                            "their",
                        }
                    ):
                        seen_word = True
                        i += 1
                        continue
                if not _allow_prepositional_split(tokens, i):
                    seen_word = True
                    i += 1
                    continue
                after = sum(1 for t2 in tokens[i + 1 : i + 8] if _is_word(t2))
                if after >= 2:
                    splits.append(i)
            seen_word = True
        i += 1
    return splits


def _label_remainder(tokens: list[str]) -> str:
    """Classify a leftover chunk as main / noun / verb phrase."""
    words = [t for t in tokens if _is_word(t)]
    if not words:
        return MAIN_CLAUSE
    lows = [_norm(w) for w in words]
    if lows and lows[0] in _COORD:
        # Coordinated continuation — keep as main reading chunk color.
        return MAIN_CLAUSE
    if any(w in _AUX for w in lows) or any(
        w.endswith("s") or w.endswith("ed") or w.endswith("ing") for w in lows[1:2]
    ):
        if lows[0] in _AUX or lows[0] in {"i", "you", "he", "she", "it", "we", "they"}:
            return MAIN_CLAUSE
        if any(w in _AUX for w in lows):
            return MAIN_CLAUSE
        return VERB_PHRASE
    if lows[0] in {"the", "a", "an", "this", "that", "these", "those", "my", "your"}:
        return NOUN_PHRASE
    return MAIN_CLAUSE


def _segment_sentence(sentence: str) -> list[GrammarSegment]:
    """Split one sentence into learner-friendly reading chunks."""
    sentence = sentence.strip()
    if not sentence:
        return []
    tokens = _tokenize(sentence)
    if not tokens:
        return [{"text": sentence, "label": MAIN_CLAUSE}]

    clause_cuts: list[tuple[int, str]] = []
    initial_head_label: str | None = None

    initial = _find_sentence_initial_phrase_end(tokens)
    if initial is not None:
        main_start, initial_head_label = initial
        clause_cuts.append((main_start, MAIN_CLAUSE))

    rel = _find_relative_split(tokens)
    if rel is not None and all(rel != c for c, _ in clause_cuts):
        clause_cuts.append((rel, RELATIVE_CLAUSE))
        rel_end = _relative_clause_end(tokens, rel)
        if rel_end < len(tokens) and all(rel_end != c for c, _ in clause_cuts):
            clause_cuts.append((rel_end, MAIN_CLAUSE))

    dep = _find_dependent_split(tokens)
    if dep is not None and all(dep != c for c, _ in clause_cuts):
        clause_cuts.append((dep, DEPENDENT_CLAUSE))

    part = _find_participial_split(tokens)
    if part is not None and all(part != c for c, _ in clause_cuts):
        clause_cuts.append((part, PARTICIPIAL_PHRASE))

    inf = _find_infinitive_split(tokens)
    if inf is not None and all(inf != c for c, _ in clause_cuts):
        inside_relative = False
        if rel is not None:
            rel_end = _relative_clause_end(tokens, rel)
            if rel <= inf < rel_end:
                inside_relative = True
        if not inside_relative:
            clause_cuts.append((inf, INFINITIVE_PHRASE))

    cuts: list[tuple[int, str]] = list(clause_cuts)

    # Prepositional phrases (may be several, spaced apart).
    if initial_head_label is None:
        last_cut = -1
        for p in _find_prepositional_splits(tokens):
            if any(p == c for c, _ in cuts):
                continue
            # Keep PP breaks from clustering too tightly.
            if last_cut >= 0 and (p - last_cut) < 4:
                continue
            cuts.append((p, PREPOSITIONAL_PHRASE))
            last_cut = p

    # Comma list / pause breaks — important for sentences without which/because.
    for p in _find_comma_pause_splits(tokens):
        if any(p == c for c, _ in cuts):
            continue
        cuts.append((p, NOUN_PHRASE))

    # Coordinating and/but/or between substantial chunks.
    for p in _find_coord_splits(tokens):
        if any(p == c for c, _ in cuts):
            continue
        cuts.append((p, MAIN_CLAUSE))

    if not cuts:
        return [{"text": _join_tokens(tokens), "label": _label_remainder(tokens)}]

    cuts.sort(key=lambda x: x[0])
    seen: set[int] = set()
    unique: list[tuple[int, str]] = []
    for idx, label in cuts:
        if idx in seen or idx <= 0 or idx >= len(tokens):
            continue
        # Drop cuts that are too close to the previous kept cut (but allow
        # list-item commas that are ~3 tokens apart: \"A, B, C\").
        if unique and (idx - unique[-1][0]) < 2:
            continue
        seen.add(idx)
        unique.append((idx, label))
    unique.sort(key=lambda x: x[0])

    if not unique:
        return [{"text": _join_tokens(tokens), "label": _label_remainder(tokens)}]

    segments: list[GrammarSegment] = []
    start = 0
    open_label: str | None = None
    for idx, label in unique:
        chunk = tokens[start:idx]
        if chunk:
            text = _join_tokens(chunk)
            if text:
                if open_label is None:
                    lab = initial_head_label or _label_remainder(chunk)
                else:
                    lab = open_label
                segments.append({"text": text, "label": lab})
        start = idx
        open_label = label

    if start < len(tokens):
        tail = tokens[start:]
        text = _join_tokens(tail)
        if text:
            lab = open_label or _label_remainder(tail)
            segments.append({"text": text, "label": lab})

    cleaned: list[GrammarSegment] = []
    for seg in segments:
        t = seg["text"].strip()
        if not t:
            continue
        # Punctuation-only scraps stay attached to the previous chunk.
        if cleaned and not any(c.isalnum() for c in t):
            cleaned[-1]["text"] = _glue_texts(cleaned[-1]["text"], t)
            continue
        words = t.split()
        # Only glue true scraps (a single short word), never 2-word list items.
        if cleaned and len(words) == 1 and len(t) <= 4:
            cleaned[-1]["text"] = _glue_texts(cleaned[-1]["text"], t)
            continue
        cleaned.append({"text": t, "label": seg["label"]})
    return cleaned or [{"text": sentence, "label": MAIN_CLAUSE}]


def _word_count(text: str) -> int:
    return sum(1 for t in text.split() if any(c.isalnum() for c in t))


def _merge_short_segments(
    segments: list[GrammarSegment], min_words: int = MIN_CHUNK_WORDS
) -> list[GrammarSegment]:
    """Merge only tiny scraps so real reading breaks stay visible.

    Previous logic kept absorbing every following chunk into a short opener,
    which erased all · marks on sentences like \"I also worked on …\".
    Now: a short chunk merges with the *next* chunk once, then we continue.
    """
    # Forward-merge short → next (keep next label when next is longer).
    forward: list[GrammarSegment] = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        label = seg.get("label", MAIN_CLAUSE)
        if label in _BREAK_LABELS:
            forward.append(seg)
            i += 1
            continue
        piece = (seg.get("text") or "").strip()
        if not piece:
            i += 1
            continue
        w = _word_count(piece)
        if (
            w < min_words
            and i + 1 < len(segments)
            and segments[i + 1].get("label") not in _BREAK_LABELS
        ):
            nxt = segments[i + 1]
            nxt_text = (nxt.get("text") or "").strip()
            if nxt_text:
                nxt_w = _word_count(nxt_text)
                # Prefer the longer chunk's grammatical color.
                use_label = nxt.get("label", label) if nxt_w >= w else label
                forward.append(
                    {"text": _glue_texts(piece, nxt_text), "label": use_label}
                )
                i += 2
                continue
        forward.append({"text": piece, "label": label})
        i += 1

    # Backward-merge: trailing scraps into previous reading chunk.
    out: list[GrammarSegment] = []
    for seg in forward:
        label = seg.get("label", MAIN_CLAUSE)
        if label in _BREAK_LABELS:
            out.append(seg)
            continue
        piece = (seg.get("text") or "").strip()
        if not piece:
            continue
        w = _word_count(piece)
        if (
            out
            and out[-1]["label"] not in _BREAK_LABELS
            and w < min_words
        ):
            out[-1]["text"] = _glue_texts(out[-1]["text"], piece)
            continue
        out.append({"text": piece, "label": label})
    return out


def parse_grammar(text: str) -> list[GrammarSegment]:
    """Segment English text into labeled reading chunks for learners.

    Paragraph / sentence breaks are preserved so the UI can stack chunks.
    Short chunks are merged so break points are comfortably spaced.
    """
    text = (text or "").strip()
    if not text:
        return []

    paragraphs = re.split(r"\n\s*\n", text)
    all_segments: list[GrammarSegment] = []

    for pi, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue
        para_flat = re.sub(r"\s*\n\s*", " ", para)
        sentences = _SENTENCE_SPLIT.split(para_flat)
        for si, sent in enumerate(sentences):
            sent = sent.strip()
            if not sent:
                continue
            segs = _segment_sentence(sent)
            all_segments.extend(segs)
            if si < len(sentences) - 1 and segs:
                all_segments.append({"text": "", "label": "sentence_break"})
        if pi < len(paragraphs) - 1 and all_segments:
            all_segments.append({"text": "", "label": "paragraph_break"})

    if not all_segments:
        return [{"text": text, "label": MAIN_CLAUSE}]
    return _merge_short_segments(all_segments)


def format_with_marks(text: str, mark: str = " · ") -> str:
    """Plain-text preview with separators (useful for tests / debugging)."""
    parts: list[str] = []
    for seg in parse_grammar(text):
        if seg["label"] in {"paragraph_break", "sentence_break"}:
            parts.append("\n")
            continue
        if parts and parts[-1] not in {"\n"}:
            parts.append(mark)
        parts.append(seg["text"])
    return "".join(parts).strip()
