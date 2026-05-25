#!/usr/bin/env python3
"""
Hard-constrained CTC decoding for shabad follow-along.

Unlike pyctcdecode (soft lexicon bias), this FORCES the decoder to only
produce words that exist in the locked shabad's vocabulary.

Approach:
  1. Tokenize every shabad word → SentencePiece token sequences
  2. Build a prefix trie of all valid token sequences
  3. At each CTC frame, mask logprobs to -inf for tokens that can't continue
     any valid word prefix (blank always allowed)
  4. Greedy argmax on the masked logprobs

This guarantees every decoded word is in the shabad — no fuzzy matching needed
for character-level ASR errors like ਫ→ਭ.

Usage:
    python scripts/prototype_hard_constrained_ctc.py --track-id 94507
    python scripts/prototype_hard_constrained_ctc.py --track-id 94507 --output-dir logs/windowed_transcriptions_2s_hard
"""

import re
import sys
from pathlib import Path

import numpy as np
# torch is imported lazily inside get_model() / extract_logprobs() so callers
# that only need the trie + decode helpers (shabad_engine in ONNX
# mode, the desktop sidecar bundle) don't pay the ~200 MB import cost or
# need torch on the install path. The hot-loop decoders below
# (greedy_decode_with_timestamps, hard_constrained_decode, build_trie,
# load_shabad_lines, build_shabad_lexicon, normalize_gurmukhi) are pure
# numpy + Python and don't touch torch.

# ============================================================================
# Paths & constants
# ============================================================================
# Repo root resolver — same pattern as shabad_engine.py: in a
# PyInstaller bundle, sys._MEIPASS holds the data/ directory; otherwise
# walk up from this file. Added 2026-05-05 for the desktop sidecar.
_REPO_ROOT = Path(getattr(sys, "_MEIPASS", None) or Path(__file__).resolve().parent.parent)
SAMPLE_RATE = 16000

# Frame duration for IndicConformer CTC output
FRAME_DURATION_S = 0.08

# Token indices
BLANK_ID = 256  # "|" in vocab
UNK_ID = 0      # "<unk>" in vocab


# ============================================================================
# Text normalization + shabad loading (same as generate_constrained_transcriptions.py)
# ============================================================================

def normalize_gurmukhi(text: str) -> str:
    text = re.sub(r'[॥੦੧੨੩੪੫੬੭੮੯।]', '', text)
    text = re.sub(r'([\u0a3e-\u0a4d\u0a70\u0a71])\1+', r'\1', text)
    return ' '.join(text.split()).strip()


def get_verse_unicode(verse: dict) -> str:
    gurmukhi = verse.get("gurmukhi", "")
    if isinstance(gurmukhi, dict):
        spaced = gurmukhi.get("unicode", "")
        if spaced and " " in spaced:
            return spaced
    return verse.get("unicode", "")


_HEADER_PATTERNS = [
    r'^\s*ੴ\s*ਸਤਿਗੁਰ\s*ਪ੍ਰਸਾਦਿ\s*$',
    r'^\s*(ਸਲੋਕੁ|ਪਉੜੀ|ਛੰਤ|ਅਸਟਪਦੀ|ਸੋਹਿਲਾ)\s*$',
    r'^\s*(ਰਾਗੁ|ਬਿਲਾਵਲੁ|ਗਉੜੀ|ਸੋਰਠਿ|ਸਿਰੀਰਾਗੁ|ਮਾਝ|ਆਸਾ|ਗੂਜਰੀ|ਧਨਾਸਰੀ|ਟੋਡੀ|ਤਿਲੰਗ|ਸੂਹੀ|ਰਾਮਕਲੀ|ਮਾਰੂ|ਤੁਖਾਰੀ|ਕੇਦਾਰਾ|ਭੈਰਉ|ਬਸੰਤੁ|ਸਾਰੰਗ|ਮਲਾਰ|ਕਾਨੜਾ|ਕਲਿਆਣ|ਪ੍ਰਭਾਤੀ|ਜੈਤਸਰੀ|ਵਡਹੰਸੁ)\s+ਮਹਲਾ',
    r'^\s*ਮਃ\s*[੧੨੩੪੫]\s*$',
    r'^\s*ਮਹਲਾ\s*[੧੨੩੪੫]\s*$',
]
_HEADER_RE = [re.compile(p) for p in _HEADER_PATTERNS]


def is_header_line(text: str) -> bool:
    return any(r.match(text) for r in _HEADER_RE)


def load_shabad_lines(verses: list[dict]) -> list[dict]:
    """Convert corpus verse dicts into the format needed for trie building.

    Accepts the list returned by ShabadCorpus.get_lines(shabad_id).
    """
    lines = []
    for i, verse in enumerate(verses):
        text = get_verse_unicode(verse)
        normalized = normalize_gurmukhi(text)
        if len(normalized) < 5 or is_header_line(normalized):
            continue
        lines.append({"index": i, "text": text, "normalized": normalized})
    return lines


def build_shabad_lexicon(lines: list[dict]) -> list[str]:
    words = set()
    for line in lines:
        for w in line["normalized"].split():
            if len(w) >= 2:
                words.add(w)
    return sorted(words)


# ============================================================================
# SentencePiece tokenization (greedy longest-match using the vocab list)
# ============================================================================

def tokenize_word(word: str, vocab: list[str]) -> list[int]:
    """Tokenize a Gurmukhi word into SentencePiece token indices.
    
    Adds ▁ prefix (SentencePiece word boundary) and does greedy
    longest-prefix matching against the vocab.
    """
    text = "▁" + word
    pos = 0
    tokens = []
    while pos < len(text):
        best_len = 0
        best_idx = -1
        for i, t in enumerate(vocab):
            if i == BLANK_ID:
                continue
            tlen = len(t)
            if tlen > best_len and text[pos:pos + tlen] == t:
                best_len = tlen
                best_idx = i
        if best_idx >= 0:
            tokens.append(best_idx)
            pos += best_len
        else:
            # Character not in vocab — use <unk> and skip one char
            tokens.append(UNK_ID)
            pos += 1
    return tokens


# ============================================================================
# Prefix trie for hard-constrained CTC
# ============================================================================

class TokenTrie:
    """Prefix trie over SentencePiece token sequences.
    
    Each path from root to a terminal node represents one valid word's
    token sequence. The trie tells us which token IDs are valid
    continuations at any given prefix state.
    
    State is represented as a tuple of (node_id,) which allows tracking
    multiple simultaneous positions in the trie.
    """
    
    def __init__(self):
        # Each node: {token_id → child_node_id}
        self.nodes = [{}]     # node 0 = root
        self.terminal = {0}   # root is terminal (we can always start a new word)
        self.words = {}       # node_id → word string (for terminals)
        self.words[0] = ""    # root = between words
        self._next_id = 1
    
    def add_word(self, token_ids: list[int], word: str):
        """Add a word's token sequence to the trie."""
        node = 0
        for tid in token_ids:
            if tid not in self.nodes[node]:
                self.nodes.append({})
                self.nodes[node][tid] = self._next_id
                self._next_id += 1
            node = self.nodes[node][tid]
        self.terminal.add(node)
        self.words[node] = word
    
    def valid_tokens_from(self, node: int) -> set[int]:
        """Return set of token IDs that are valid transitions from this node."""
        return set(self.nodes[node].keys())
    
    def advance(self, node: int, token_id: int) -> int:
        """Advance to child node. Returns -1 if invalid."""
        return self.nodes[node].get(token_id, -1)
    
    def is_terminal(self, node: int) -> bool:
        return node in self.terminal
    
    def valid_tokens_at_state(self, states: set[int]) -> set[int]:
        """Given a set of active trie states, return all valid next tokens.
        
        This is the key method for CTC masking:
        - From each active state, collect valid continuation tokens
        - If any state is terminal, also include word-start tokens (from root)
        """
        valid = set()
        for s in states:
            valid.update(self.valid_tokens_from(s))
            # If this state is terminal (complete word), we can start a new word
            if self.is_terminal(s):
                valid.update(self.valid_tokens_from(0))  # root transitions
        return valid
    
    def advance_states(self, states: set[int], token_id: int) -> set[int]:
        """Advance all active states by the given token. Return new state set.

        Multi-word support: if a state is terminal (complete word), we also
        try starting a new word from root. This lets the beam cross word
        boundaries within the shabad's lexicon without requiring an explicit
        word-separator token.
        """
        new_states = set()
        for s in states:
            nxt = self.advance(s, token_id)
            if nxt >= 0:
                new_states.add(nxt)
            # If this state is terminal and token starts a new word from root
            if self.is_terminal(s):
                nxt2 = self.advance(0, token_id)
                if nxt2 >= 0:
                    new_states.add(nxt2)
        return new_states


def build_trie(words: list[str], vocab: list[str]) -> TokenTrie:
    """Build a token trie from a list of Gurmukhi words."""
    trie = TokenTrie()
    tokenized = 0
    failed = 0
    for word in words:
        tokens = tokenize_word(word, vocab)
        if UNK_ID in tokens:
            failed += 1
            continue
        trie.add_word(tokens, word)
        tokenized += 1
    return trie


# ============================================================================
# Hard-constrained BEAM SEARCH CTC decode
# ============================================================================

# Beam search width
BEAM_WIDTH = 10


def hard_constrained_decode(
    logprobs: np.ndarray,
    vocab: list[str],
    trie: TokenTrie,
    beam_width: int = BEAM_WIDTH,
) -> tuple[str, list[dict]]:
    """
    Beam-search CTC decode with HARD trie constraint.
    
    Unlike greedy, beam search explores multiple paths simultaneously,
    so it won't get stuck when the locally-best token leads to a dead end.
    
    Example: at a frame where P(▁ਫ)=-0.0 and P(▁ਭ)=-4.9:
      - Greedy picks ▁ਫ, gets stuck (ਫੂ... not in shabad)
      - Beam search keeps BOTH paths; when ▁ਫ path stalls, ▁ਭ wins
    
    Each beam = (cumulative_logprob, trie_states: frozenset, 
                 token_sequence: list[(frame, token_id)], last_token: int)
    
    CTC rules:
      - blank: keep beam alive, don't advance trie, don't emit
      - repeat of last_token: keep beam alive, don't emit
      - new token: emit, advance trie
    
    Beams are keyed by (trie_states, last_token) for merging. When two beams
    reach the same (state, last_tok), we keep the one with higher logprob.
    
    Returns (text, timestamps) in the same format as greedy_decode_with_timestamps.
    """
    T, V = logprobs.shape
    
    # beam: (cum_logprob, frozenset_of_trie_states, emitted_tokens, last_token)
    initial_beam = (0.0, frozenset({0}), [], BLANK_ID)
    beams = [initial_beam]
    
    for t in range(T):
        frame_lp = logprobs[t]
        # Key: (trie_states, last_tok) → (cum_lp, states, tokens, last_tok)
        new_beams = {}
        
        for cum_lp, states, tokens, last_tok in beams:
            # 1. BLANK transition
            blank_lp = cum_lp + float(frame_lp[BLANK_ID])
            bkey = (states, BLANK_ID)
            if bkey not in new_beams or new_beams[bkey][0] < blank_lp:
                new_beams[bkey] = (blank_lp, states, tokens, BLANK_ID)
            
            # 2. REPEAT of last token (non-blank)
            if last_tok != BLANK_ID:
                rep_lp = cum_lp + float(frame_lp[last_tok])
                rkey = (states, last_tok)
                if rkey not in new_beams or new_beams[rkey][0] < rep_lp:
                    new_beams[rkey] = (rep_lp, states, tokens, last_tok)
            
            # 3. NEW token emissions
            valid_tokens = trie.valid_tokens_at_state(set(states))
            for tok in valid_tokens:
                if tok == BLANK_ID or tok >= V or tok == last_tok:
                    continue
                
                tok_lp = cum_lp + float(frame_lp[tok])
                new_states = trie.advance_states(set(states), tok)
                if not new_states:
                    continue
                
                # If terminal state reached, also allow starting new word
                for s in list(new_states):
                    if trie.is_terminal(s):
                        new_states.add(0)
                        break
                
                fs = frozenset(new_states)
                new_toks = tokens + [(t, tok)]
                ekey = (fs, tok)
                if ekey not in new_beams or new_beams[ekey][0] < tok_lp:
                    new_beams[ekey] = (tok_lp, fs, new_toks, tok)
        
        # Prune to top beam_width
        candidates = list(new_beams.values())
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]
        
        if not beams:
            beams = [(0.0, frozenset({0}), [], BLANK_ID)]
    
    # Pick best beam
    best = max(beams, key=lambda x: x[0])
    _, _, raw_tokens, _ = best
    
    return _tokens_to_text(raw_tokens, vocab)


def _tokens_to_text(
    raw_tokens: list[tuple[int, int]],
    vocab: list[str],
) -> tuple[str, list[dict]]:
    """Convert (frame_idx, token_id) sequence to text + word timestamps."""
    words = []
    current_word_tokens = []
    current_word_start = None
    current_word_end = None
    
    for frame_idx, tok_id in raw_tokens:
        tok_str = vocab[tok_id] if tok_id < len(vocab) else "?"
        
        # ▁ prefix means word boundary — emit previous word and start new one
        if tok_str.startswith("▁") and current_word_tokens:
            word_text = "".join(current_word_tokens)
            if word_text.strip():
                words.append({
                    "word": word_text,
                    "start": round(current_word_start * FRAME_DURATION_S, 3),
                    "end": round(current_word_end * FRAME_DURATION_S, 3),
                })
            current_word_tokens = [tok_str[1:]]  # strip ▁
            current_word_start = frame_idx
            current_word_end = frame_idx
        else:
            if current_word_start is None:
                current_word_start = frame_idx
                if tok_str.startswith("▁"):
                    current_word_tokens.append(tok_str[1:])
                else:
                    current_word_tokens.append(tok_str)
            else:
                current_word_tokens.append(tok_str)
            current_word_end = frame_idx
    
    # Emit last word
    if current_word_tokens:
        word_text = "".join(current_word_tokens)
        if word_text.strip():
            words.append({
                "word": word_text,
                "start": round(current_word_start * FRAME_DURATION_S, 3),
                "end": round(current_word_end * FRAME_DURATION_S, 3),
            })
    
    text = " ".join(w["word"] for w in words)
    return text, words


# ============================================================================
# Standard greedy decode (for comparison)
# ============================================================================

def greedy_decode_with_timestamps(
    logprobs: np.ndarray,
    vocab: list[str],
    blank_id: int = 256,
) -> tuple[str, list[dict]]:
    """Greedy decode with word-level timestamps from frame positions."""
    T, V = logprobs.shape
    token_ids = np.argmax(logprobs, axis=1)

    prev = blank_id
    raw = []
    for t, tok_idx in enumerate(token_ids):
        if tok_idx != blank_id and tok_idx != prev:
            char = vocab[tok_idx]
            raw.append((t, char))
        prev = tok_idx

    # Group into words (split on ▁ which is the word boundary in SentencePiece)
    words = []
    current_chars = []
    start_frame = None
    end_frame = None
    for frame, char in raw:
        if char.startswith("▁") and current_chars:
            word = "".join(current_chars)
            if word.strip():
                words.append({
                    "word": word,
                    "start": round(start_frame * FRAME_DURATION_S, 3),
                    "end": round(end_frame * FRAME_DURATION_S, 3),
                })
            current_chars = [char[1:]]
            start_frame = frame
            end_frame = frame
        else:
            if start_frame is None:
                start_frame = frame
            current_chars.append(char.replace("▁", ""))
            end_frame = frame

    if current_chars:
        word = "".join(current_chars)
        if word.strip():
            words.append({
                "word": word,
                "start": round(start_frame * FRAME_DURATION_S, 3),
                "end": round(end_frame * FRAME_DURATION_S, 3),
            })

    text = " ".join(w["word"] for w in words)
    return text, words


# ============================================================================
# Compare: side-by-side analysis
# ============================================================================
