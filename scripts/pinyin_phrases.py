"""Segment and annotate public corpus text with the fixed phrase supplement."""
import hashlib
from importlib.metadata import version
import json
import re

import jieba
from pypinyin import Style, lazy_pinyin, load_phrases_dict

PYPINYIN_VERSION = "0.55.0"
JIEBA_VERSION = "0.42.1"
_tokenizer = jieba.Tokenizer()


def load_pinyin_phrases(path):
    global _tokenizer
    if version("pypinyin") != PYPINYIN_VERSION:
        raise ValueError(f"pypinyin must be exactly {PYPINYIN_VERSION}")
    if version("jieba") != JIEBA_VERSION:
        raise ValueError(f"jieba must be exactly {JIEBA_VERSION}")
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
    _tokenizer = jieba.Tokenizer()
    for phrase in phrases:
        _tokenizer.add_word(phrase)
    return hashlib.sha256(raw).hexdigest()


def syllables(text):
    tokens = list(_tokenizer.cut(text, cut_all=False, HMM=True))
    values = lazy_pinyin(tokens, style=Style.NORMAL, strict=True, errors=list)
    if len(values) != len(text):
        raise ValueError("pypinyin did not preserve character alignment")
    return values
