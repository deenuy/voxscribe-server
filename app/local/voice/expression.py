"""How the assistant sounds: the spoken register, and every list of words it may say on its own.

Every list of spoken strings this register uses lives in this module, by name: the reactions a
reply may open with, the thinking lines, the farewells, and the pronunciation table. Strike a
name from a list here and it is gone everywhere, the audition included
(`python -m app.local.voice.audition`). Logs name list entries (`reaction=oh`,
`thinking=let_me_see`, `farewell=talk_soon`), never the words a person or the model said.

The setting (VOXSCRIBE_VOICE_EXPRESSIVENESS):
  plain    today's replies: the engine's prompt, the reply model's words as it wrote them
  natural  a spoken register: the prompt asks for speech, and the rules below hold on what the
           reply model wrote, deterministically, before the gate
The default comes from the product mode the gate runs under: natural for personal and
IdeaScribe, plain for senior care and classroom (warmth that invites attachment is a question
for a clinician and an education specialist, not for this module).

The rules (models propose, the deterministic controller disposes). They apply to the reply
model's own sentences only, never to a refusal, a safety line, a read-back, a confirmation or
any other scripted line:
  - at most one reaction per reply, only at its start, only from REACTIONS (the interjections
    our voice pronounces, measured on its phonemizer), and not the one the previous reply in the
    session opened with; any other interjection is removed
  - an acknowledgement opener ("Sure,", "Of course,") the previous reply also opened with is
    removed, so two replies running never open the same way
  - stock assistant sentences (STOCK_LEADING) are removed when they lead the reply, and closing
    offers (STOCK_OFFERS) wherever they stand; a reply made of nothing else keeps its last one
  - hesitation fillers between commas (", um,") are removed
Replies arrive sentence by sentence; the rules work on that stream and hold nothing but a stock
sentence they are about to remove.

Pronunciation (`pronounce`) runs after the gate, immediately before synthesis: what the gate
passed, the captions and the stored transcript keep the original spelling.
"""
from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

PLAIN = "plain"
NATURAL = "natural"
EXPRESSIVENESS = (PLAIN, NATURAL)
NATURAL_MODES = frozenset({"personal", "ideascribe"})


def default_expressiveness(mode: str | None) -> str:
    """natural for personal and IdeaScribe; plain for everything else (senior care, classroom,
    and a mode this module does not know)."""
    return NATURAL if (mode or "") in NATURAL_MODES else PLAIN


def resolve_expressiveness(setting: str | None, mode: str | None) -> str:
    """The operator's setting when it names one, else the product mode's default."""
    value = (setting or "").strip().lower()
    return value if value in EXPRESSIVENESS else default_expressiveness(mode)


# ── The lists: name -> words ────────────────────────────────────────────────

# Reactions a reply may open with, as the audition says them alone. From the interjections the
# voice pronounces as a person expects (measured 2026-11-19 on Kokoro's phonemizer: Hmm, Ah,
# Aha, Oh, Ooh, Huh, Ha, Haha, Whoa, Wow, Oops, Phew). Yeah, Yep, Uh-huh, Um and Uh read well too
# but are not reactions: the first three are answers and are never touched, the last two are
# hesitations and are removed.
REACTIONS: dict[str, str] = {
    "oh": "Oh.",
    "ah": "Ah.",
    "aha": "Aha!",
    "hmm": "Hmm.",
    "ooh": "Ooh.",
    "huh": "Huh.",
    "ha": "Ha!",
    "haha": "Haha!",
    "whoa": "Whoa!",
    "wow": "Wow!",
    "oops": "Oops.",
    "phew": "Phew.",
}

# Said once when a reply is late (PL67): none when the reply starts in time.
THINKING: dict[str, str] = {
    "let_me_see": "Let me see.",
    "one_moment": "One moment.",
    "just_a_second": "Just a second.",
    "okay_so": "Okay, so...",
}

# Said once when the person closes the conversation (PL61), in natural.
FAREWELLS: dict[str, str] = {
    "anytime": "Anytime.",
    "talk_soon": "Talk soon.",
    "bye_for_now": "Bye for now.",
    "take_care": "Take care.",
    "see_you": "See you later.",
}

# The one farewell in plain.
PLAIN_FAREWELL: dict[str, str] = {"goodbye": "Goodbye."}


def farewells(expressiveness: str) -> dict[str, str]:
    return FAREWELLS if expressiveness == NATURAL else PLAIN_FAREWELL


def pick(options: dict[str, str], last: str | None) -> str | None:
    """A name from `options` that is not `last` (unless it is the only one): the one after it in
    list order, so the lines vary and never repeat twice running. None when the list is empty."""
    names = list(options)
    if not names:
        return None
    if last in names:
        return names[(names.index(last) + 1) % len(names)]
    return names[0]


# ── Pronunciation: after the gate, before synthesis ─────────────────────────

@dataclass(frozen=True)
class Pronunciation:
    """A spelling the voice misreads, and one it reads acceptably. Phonemes measured on
    Kokoro's phonemizer (en-us), 2026-11-19."""

    name: str
    pattern: re.Pattern
    replacement: str
    before: str            # what the phonemizer makes of the misread spelling
    after: str             # and of the replacement


PRONUNCIATIONS: tuple[Pronunciation, ...] = (
    # Mm-hmm, Mhm, Mmhmm, Mhmm, Mm-hm, Mmm-hmm, Mm hmm: an "um hum", the agreeing hum.
    Pronunciation("mm_hmm", re.compile(r"\b[Mm]+(?:[- ]?[Hh]+[Mm]+)\b"), "Um-hum",
                  before="ˌɛmˈɛmhəm (em em hum); ˌɛmˌeɪtʃˈɛm (em aitch em)",
                  after="ˈʌmhˈʌm"),
    # Mm, Mmm: the neighbour Hmm. Capitalised only, so the unit "mm" is never touched.
    Pronunciation("mm", re.compile(r"\bMm+\b"), "Hmm",
                  before="ˌɛmˈɛm (em em)", after="həm"),
    # Hm and Hmmm...: spelled out as letters when the m is not doubled exactly.
    Pronunciation("hm", re.compile(r"\b([Hh])m(?:m{2,})?\b"), r"\1mm",
                  before="ˌeɪtʃˈɛm (aitch em)", after="həm"),
    # Aww: "aw-uh".
    Pronunciation("aww", re.compile(r"\b([Aa])w{2,}\b"), r"\1w",
                  before="ˈɔːwə (aw-uh)", after="ˈɔː"),
    # Wowww: "wow-wuh".
    Pronunciation("wowww", re.compile(r"\b([Ww])ow{2,}\b"), r"\1ow",
                  before="wˈaʊʊwə", after="wˈaʊ"),
)


def pronounce(text: str) -> str:
    """The text the voice is given: each entry of PRONUNCIATIONS rewritten. Never shown or
    stored."""
    for entry in PRONUNCIATIONS:
        text = entry.pattern.sub(entry.replacement, text)
    return text


# ── Reading the reply model's sentences ─────────────────────────────────────

# Interjection words as the model writes them, with stretched spellings, to their canonical
# name. Allowed names are REACTIONS; the rest are recognised only to be removed.
_INTERJECTIONS: tuple[tuple[str, re.Pattern], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE)) for name, pattern in (
        ("haha", r"h+a+(?:h+a+)+h*"),
        ("aha", r"a+h+a+"),
        ("ooh", r"o{2,}h+"),
        ("oh", r"oh+"),
        ("ah", r"a+h+"),
        ("hmm", r"h+m+"),
        ("huh", r"h+u+h+"),
        ("ha", r"ha+h*"),
        ("whoa", r"w+h+o+a+"),
        ("wow", r"w+o+w+"),
        ("oops", r"o+p+s+"),
        ("phew", r"p+h+e+w+"),
        ("mm_hmm", r"m+(?:-?h+m+)"),
        ("uh_huh", r"u+h+-h+u+h+"),
        ("mm", r"m{2,}"),
        ("um", r"u+m+"),
        ("uh", r"u+h+"),
        ("er", r"e+r+m*"),
        ("aw", r"a+w+"),
        ("ugh", r"u+g+h+"),
        ("yay", r"y+a+y+"),
        ("gosh", r"gosh"),
    ))
# Uh-huh is an answer when it stands alone; it is removed only as a second interjection.
_WORD = re.compile(r"\s*([A-Za-z]+(?:-[A-Za-z]+)?)([,.!?…]*)(?=\s|$)")

ACK_OPENERS = ("of course", "all right", "sure", "okay", "ok", "alright", "absolutely",
               "certainly", "great", "well", "so")
_ACK = re.compile(r"^\s*(" + "|".join(re.escape(a) for a in ACK_OPENERS) + r")\s*,\s*(\S.*)$",
                  re.IGNORECASE | re.DOTALL)

# Whole sentences that lead a reply and carry nothing: removed when they stand before the answer.
STOCK_LEADING: dict[str, re.Pattern] = {name: re.compile(p, re.IGNORECASE) for name, p in {
    "thanks_for_asking": r"(?:thank you|thanks)(?: so much)? for asking",
    "here_to_help": r"(?:i'm|i am) (?:always )?here to help(?: you)?(?: with (?:that|anything))?",
    "happy_to_help": r"(?:i'd|i would|i'll|i will) be (?:happy|glad|delighted) to help(?: (?:you )?with that)?",
    "great_question": r"(?:(?:that's|that is|what) )?(?:a )?(?:great|good|wonderful|excellent|fantastic) question",
    "as_an_ai": r"as an ai(?: (?:language model|assistant))?(?:, (?:i'm|i am) here to help(?: you)?)?",
    "how_can_i_help": r"how (?:can|may) i (?:help|assist) you(?: today)?",
}.items()}
# Closing offers: removed wherever they stand, since they never carry the answer.
STOCK_OFFERS: dict[str, re.Pattern] = {name: re.compile(p, re.IGNORECASE) for name, p in {
    "anything_else": r"(?:is there )?anything else (?:i can (?:help|assist) you with|you(?:'d| would) like to (?:know|discuss|talk about))",
    "let_me_know": r"(?:just )?let me know if (?:you have any (?:other |more )?questions|(?:there's|there is) anything else(?: i can (?:do|help with))?|you need anything(?: else)?)",
    "feel_free": r"feel free to ask(?: me)?(?: (?:anything|any other questions))?",
    "how_can_i_help": STOCK_LEADING["how_can_i_help"].pattern,
}.items()}
# A clause of thanks at either end of a sentence: "I'm well, thanks for asking."
_THANKS_CLAUSE_END = re.compile(r",\s*(?:thank you|thanks)(?: so much)? for asking\s*([.!?]*)\s*$",
                                re.IGNORECASE)
_THANKS_CLAUSE_START = re.compile(r"^\s*(?:thank you|thanks)(?: so much)? for asking\s*,\s*(\S)",
                                  re.IGNORECASE)
# A closing clause offering help: "Yes, I can hear you, and I'm ready to assist you with ..."
_HELP_CLAUSE_END = re.compile(r",\s*and (?:i'm|i am) (?:here|ready|happy|glad|always here) to "
                              r"(?:help|assist)\b[^.!?]*([.!?]*)\s*$", re.IGNORECASE)
_HESITATION_CLAUSE = re.compile(r",\s*(?:u+m+|u+h+|e+r+m*)\s*,", re.IGNORECASE)


def _canonical(word: str) -> str | None:
    for name, pattern in _INTERJECTIONS:
        if pattern.fullmatch(word):
            return name
    return None


def _bare(sentence: str) -> str:
    return re.sub(r"[\s.!?…,]+$", "", sentence.strip()).replace("’", "'")


def _is(patterns: dict[str, re.Pattern], sentence: str) -> str | None:
    bare = _bare(sentence)
    for name, pattern in patterns.items():
        if pattern.fullmatch(bare):
            return name
    return None


def _capitalise(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


@dataclass
class Opening:
    """The leading interjections of a sentence and what follows them."""

    names: list[str]
    punctuation: str
    rest: str


def split_interjections(sentence: str) -> Opening:
    names: list[str] = []
    punct = ""
    pos = 0
    while True:
        m = _WORD.match(sentence, pos)
        if not m:
            break
        name = _canonical(m.group(1))
        if name is None:
            break
        if name == "uh_huh" and not names and not sentence[m.end():].strip():
            break                       # "Uh-huh." alone is an answer
        names.append(name)
        punct = punct or m.group(2)
        pos = m.end()
        if m.group(2) and m.group(2)[-1] in ".!?…":
            break                       # the interjection is a sentence of its own
    return Opening(names, punct, sentence[pos:].strip())


def _tidy(text: str) -> str:
    text = _HESITATION_CLAUSE.sub(",", text)
    text = _THANKS_CLAUSE_START.sub(lambda m: m.group(1).upper(), text)
    for clause in (_THANKS_CLAUSE_END, _HELP_CLAUSE_END):
        m = clause.search(text.replace("’", "'"))
        if m and text[:m.start()].strip():
            text = text[:m.start()].rstrip() + (m.group(1) or ".")
    return text


@dataclass
class ReplyShape:
    """What the rules did to one reply: the opener it ended with, for the next reply's rule, and
    the names of what was removed or kept, for the log."""

    opener: str | None = None
    reaction: str | None = None
    removed: list[str] | None = None

    def note(self, name: str) -> None:
        if self.removed is None:
            self.removed = []
        self.removed.append(name)


async def shape_reply(sentences: AsyncIterator[str], *, previous_opener: str | None,
                      shape: ReplyShape, reaction_allowed: bool = True) -> AsyncIterator[str]:
    """The reply model's sentences with the natural rules applied (module docstring).

    `previous_opener` is how the previous reply in the session opened; `shape` is filled in as
    the reply goes. With `reaction_allowed` False no sentence keeps an interjection (the
    thinker's summary, delivered after a scripted line)."""
    emitted = 0
    held: str | None = None          # a stock sentence, kept only if nothing else comes
    slot_open = reaction_allowed     # the reply's one reaction, at its start
    async for raw in sentences:
        sentence = _tidy(raw)
        opening = split_interjections(sentence)
        rest = opening.rest
        stock = (_is(STOCK_LEADING, rest) if emitted == 0 else None) or _is(STOCK_OFFERS, rest)
        if stock and rest:
            shape.note(f"stock={stock}")
            if emitted == 0:
                held = sentence
            continue
        kept: str | None = None
        if opening.names:
            first = opening.names[0]
            if slot_open and first in REACTIONS and first != previous_opener:
                kept = first
            for name in opening.names:
                if name != kept:
                    shape.note(f"interjection={name}")
        slot_open = False
        if kept is not None:
            spoken = REACTIONS[kept].rstrip(".!?")
            if rest:
                mark = opening.punctuation[:1] if opening.punctuation[:1] in ",.!?" else ","
                sentence = f"{spoken}{mark} {_capitalise(rest) if mark != ',' else rest}"
            else:
                mark = opening.punctuation[:1] if opening.punctuation[:1] in ".!?" else "."
                sentence = f"{spoken}{mark}"
            shape.reaction = kept
        elif opening.names:
            if not rest:
                continue                 # a sentence that was only interjections
            sentence = _capitalise(rest)
        if emitted == 0:
            ack = _ACK.match(sentence) if kept is None else None
            if ack is not None:
                opener = ack.group(1).lower()
                if opener == previous_opener:
                    shape.note(f"opener={opener.replace(' ', '_')}")
                    sentence = _capitalise(ack.group(2))
                    opener = None
                shape.opener = opener
            else:
                shape.opener = kept
        emitted += 1
        held = None
        yield sentence
    if emitted == 0 and held is not None:
        yield held                       # the reply must say something


# ── Turns that need no thinking line ────────────────────────────────────────

BRIEF_ANSWERS = frozenset({"yes", "yeah", "yep", "yup", "sure", "okay", "ok", "correct",
                           "right", "no", "nope", "nah", "exactly", "fine"})
CORRECTION_OPENERS = ("actually", "i meant", "i said", "no i said", "no i meant", "not that",
                      "i mean")
BRIEF_ANSWER_WORDS = 3


def is_brief_answer(text: str) -> bool:
    """A turn that is only a yes, a no or a correction: no thinking line follows it."""
    norm = re.sub(r"[^\w\s']", " ", (text or "").lower()).split()
    if not norm:
        return True
    joined = " ".join(norm)
    if any(joined == c or joined.startswith(c + " ") for c in CORRECTION_OPENERS):
        return True
    return len(norm) <= BRIEF_ANSWER_WORDS and norm[0] in BRIEF_ANSWERS
