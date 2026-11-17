"""Explainable normalisation of domain slang for documents and FTS queries.

Dependencies: snowballstemmer, rapidfuzz, Unidecode. No access to chats or external services.
An alias dictionary (aliases.json) maps slang, transliterations and typos to canonical concepts.
"""
from functools import lru_cache
import json
import hashlib
from pathlib import Path
import re

from rapidfuzz.fuzz import ratio
import snowballstemmer
from unidecode import unidecode

# Separators inside compound names do not hide an alias: "py-thon-ish" still
# contains the concept. io_uring / io-uring are linked as a two-word alias.
TOKEN = re.compile(r'[a-zа-яё0-9]+(?:\+\+|#)?', re.I)
CYRILLIC = re.compile(r'^[а-яё]+$')
STOP = frozenset('а и в во на по за от до из для что как это его ему у с со ли бы о об к ко же то или the of to in a'.split())
STEMMER = snowballstemmer.stemmer('russian')
ALIASES_PATH = Path(__file__).with_name('aliases.json')
FUZZY_MIN_LENGTH = 5
FUZZY_THRESHOLD = 90
_dictionary_version = None


def refresh_dictionary():
    """Called at a query/build boundary, not for every word of the corpus."""
    global _dictionary_version
    version = hashlib.sha256(ALIASES_PATH.read_bytes()).hexdigest()
    if version != _dictionary_version:
        _dictionary.cache_clear()
        _resolve_single.cache_clear()
        _dictionary_version = version


def _plain(value):
    return value.casefold().replace('ё', 'е')


@lru_cache(maxsize=65536)
def _stem(value):
    return STEMMER.stemWord(value) if CYRILLIC.fullmatch(value) else value


@lru_cache(maxsize=65536)
def _ordinary(value):
    # FTS tokens contain only letters/digits: punctuation of the raw query is never an operator.
    return re.sub('[^a-z0-9]', '', unidecode(_stem(value)).lower())


@lru_cache(maxsize=1)
def _dictionary():
    data = json.loads(ALIASES_PATH.read_text(encoding='utf-8'))
    exact, stems, canonical = {}, {}, {}
    for concept in data['concepts']:
        name = concept['id']
        if not re.fullmatch('[a-z0-9]+', name):
            raise ValueError(f'Invalid concept id: {name}')
        canonical[name] = concept['canonical']
        for alias in concept['aliases']:
            words = tuple(_plain(word) for word in TOKEN.findall(alias))
            variants = {words, tuple(unidecode(word).lower() for word in words)}
            for form in variants:
                exact.setdefault(form, set()).add(name)
            if len(words) == 1 and len(words[0]) >= 5 and CYRILLIC.fullmatch(words[0]):
                stems.setdefault(_stem(words[0]), set()).add(name)
    return exact, stems, canonical


def _resolve(words, index, fuzzy):
    exact, stems, canonical = _dictionary()
    remaining = len(words) - index
    for count in range(min(3, remaining), 0, -1):
        fragment = tuple(words[index:index+count])
        candidates = exact.get(fragment, set())
        if len(candidates) == 1:
            return next(iter(candidates)), count, 'alias', 100
        latin = tuple(unidecode(word).lower() for word in fragment)
        candidates = exact.get(latin, set())
        if len(candidates) == 1:
            return next(iter(candidates)), count, 'transliteration', 100
    concept, method, score = _resolve_single(words[index], fuzzy)
    return concept, 1, method, score


@lru_cache(maxsize=65536)
def _resolve_single(word, fuzzy):
    exact, stems, _ = _dictionary()
    if len(word) >= 5 and CYRILLIC.fullmatch(word):
        candidates = stems.get(_stem(word), set())
        if len(candidates) == 1:
            return next(iter(candidates)), 'alias_morphology', 100
    if fuzzy and len(word) >= FUZZY_MIN_LENGTH:
        latin = unidecode(word).lower()
        # Only known aliases are considered, never arbitrary corpus words.
        scored = {}
        for alias, concepts in exact.items():
            if len(alias) != 1 or len(alias[0]) < FUZZY_MIN_LENGTH:
                continue
            target = unidecode(alias[0]).lower()
            if abs(len(latin) - len(target)) > 1:
                continue
            score = ratio(latin, target)
            if score >= FUZZY_THRESHOLD:
                for concept in concepts:
                    scored[concept] = max(score, scored.get(concept, 0))
        # Two different possible concepts: do not guess, whatever the scores.
        if len(scored) == 1:
            name, score = next(iter(scored.items()))
            return name, 'technical_typo', round(score, 2)
    return None, 'morphology', 100


def _analyze(text, fuzzy=False):
    if not isinstance(text, str):
        raise ValueError('Text must be a string')
    matches = list(TOKEN.finditer(text))
    originals = [match.group() for match in matches]
    words = [_plain(word) for word in originals]
    normalized, concepts, changes = [], [], []
    index = 0
    while index < len(words):
        word = words[index]
        if word in STOP:
            changes.append(dict(original=originals[index], normalized=None, method='stopword',
                                start=matches[index].start(), end=matches[index].end()))
            index += 1
            continue
        concept, count, method, score = _resolve(words, index, fuzzy)
        start, end = matches[index].start(), matches[index+count-1].end()
        original = text[start:end]
        if concept:
            token = 'zconcept' + concept
            if concept not in concepts:
                concepts.append(concept)
        else:
            token = _ordinary(word)
        if token:
            normalized.append(token)
        changes.append(dict(original=original, normalized=token, concept=concept,
                            method=method, score=score, start=start, end=end))
        index += count
    return normalized, concepts, changes


def normalize(text, *, fuzzy=False):
    """Normalised tokens for the index and for queries, using identical rules."""
    return _analyze(text, fuzzy=fuzzy)[0]


def semantic_text(text, *, fuzzy=True):
    """Readable text for dense retrieval / reranker: only concepts are replaced.

    Pass fuzzy=False for documents so that typo correction only ever rewrites the
    user's query, never the corpus content.
    """
    _, _, changes = _analyze(text, fuzzy=fuzzy)
    canonical = _dictionary()[2]
    pieces, cursor = [], 0
    for change in changes:
        if change.get('concept'):
            pieces.append(text[cursor:change['start']])
            pieces.append(canonical[change['concept']])
            cursor = change['end']
    pieces.append(text[cursor:])
    return ''.join(pieces)


def expand_query(query):
    """Explain every substitution. variants: original/canonical forms for dense retrieval."""
    normalized, concepts, changes = _analyze(query, fuzzy=True)
    if not normalized:
        raise ValueError('The query needs meaningful words')
    variants = list(dict.fromkeys([query, semantic_text(query)]))
    return dict(original=query, normalized=normalized, concepts=concepts,
                variants=variants, changes=changes)


def query_expression(query, all_words=False):
    """FTS MATCH built only from quoted tokens; the query is never executed as FTS syntax."""
    tokens = list(dict.fromkeys(expand_query(query)['normalized']))
    return (' AND ' if all_words else ' OR ').join('"' + token + '"' for token in tokens)
