"""Load the fixed public pronunciation supplement for corpus generation."""
import hashlib
from importlib.metadata import version
import json
import re

from pypinyin import load_phrases_dict

PYPINYIN_VERSION = "0.55.0"


def load_pinyin_phrases(path):
    if version("pypinyin") != PYPINYIN_VERSION:
        raise ValueError(f"pypinyin must be exactly {PYPINYIN_VERSION}")
    raw = path.read_bytes()
    phrases = json.loads(raw)
    if not isinstance(phrases, dict) or not phrases:
        raise ValueError("pinyin phrases must be a nonempty JSON object")
    for phrase, syllables in phrases.items():
        if (not re.fullmatch(r"[\u4e00-\u9fff]{2,8}", phrase)
                or not isinstance(syllables, list) or len(syllables) != len(phrase)
                or any(not isinstance(readings, list) or len(readings) != 1
                       or not isinstance(readings[0], str)
                       or not re.fullmatch(r"[a-z]{1,6}", readings[0]) for readings in syllables)):
            raise ValueError(f"invalid pinyin phrase: {phrase!r}; expected one lowercase syllable per Chinese character")
    load_phrases_dict(phrases)
    return hashlib.sha256(raw).hexdigest()
