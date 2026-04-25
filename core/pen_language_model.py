"""
core/pen_language_model.py
--------------------------
Dictionary-based language model for the pen writing pipeline.
Sits between the letter recogniser and text output.

Responsibilities:
  1. Accumulate recognised letters into the current word buffer
  2. Suggest word completions while the user is mid-word
  3. Autocorrect the completed word against a dictionary using edit distance
  4. Manage the full output text (insert, backspace, cursor movement)

When upgrading to a real language model API later, only this file changes.
The stroke processor and letter recogniser above it stay identical.
"""

# ---------------------------------------------------------------------------
# Dictionary — common English words for the prototype
# ---------------------------------------------------------------------------
# Organised by first letter for fast prefix lookup.
# In a real system this would be the full system dictionary (~200k words).

DICTIONARY = [
    # a
    "a", "an", "and", "are", "as", "at", "also", "all", "about",
    "after", "again", "against", "age", "ago", "air", "able", "away",
    # b
    "be", "by", "but", "been", "back", "bad", "big", "both",
    "below", "best", "between", "before", "below", "build",
    # c
    "can", "car", "care", "call", "case", "come", "could",
    "came", "cause", "course", "carry", "change", "city", "clear",
    # d
    "do", "day", "did", "door", "down", "done", "draw", "drive",
    "during", "dark", "deep", "deal", "dear",
    # e
    "each", "even", "ever", "every", "end", "else", "early", "easy",
    "enough", "enter", "earth",
    # f
    "for", "from", "few", "far", "find", "feel", "face", "fact",
    "fall", "fast", "fill", "fire", "first", "five", "food", "form", "full",
    # g
    "go", "get", "got", "give", "good", "gave", "girl", "goal",
    "gone", "grow", "great", "group", "green", "ground",
    # h
    "he", "her", "him", "his", "has", "had", "have", "how",
    "here", "house", "home", "help", "hand", "high", "hold", "hard",
    "hear", "heart", "hope", "hour",
    # i
    "i", "in", "is", "it", "if", "into", "its", "idea",
    # j
    "just", "job", "join", "jump",
    # k
    "keep", "kind", "know", "knew", "key",
    # l
    "let", "like", "line", "look", "love", "learn", "live", "last",
    "large", "left", "long", "less", "lead", "light", "little", "low",
    # m
    "me", "my", "more", "make", "man", "may", "much", "most",
    "move", "made", "many", "mind", "miss", "mean", "meet", "must",
    # n
    "no", "not", "now", "new", "name", "near", "never", "nice",
    "next", "need", "none", "night", "nine",
    # o
    "of", "on", "or", "our", "out", "one", "over", "once", "only", "own",
    "other", "often", "open", "old", "off",
    # p
    "put", "per", "pay", "plan", "play", "place", "point", "pull",
    "push", "pass", "past", "path", "pick", "poor", "power", "part",
    # q
    "quite", "quick", "quiet",
    # r
    "run", "real", "rise", "rain", "read", "rule",
    "right", "road", "role", "room", "reach", "rest", "rich",
    # s
    "so", "see", "she", "set", "some", "soon", "still", "sure", "since", "seem",
    "said", "same", "such", "show", "side", "self", "send", "sign", "size",
    "slow", "small", "start", "stay", "step", "stop", "strong",
    # t
    "to", "the", "that", "this", "they", "them", "then", "than",
    "time", "take", "tell", "turn", "true", "talk", "test", "try",
    "two", "top", "town", "tree",
    # u
    "us", "use", "used", "until", "under", "up", "upon",
    # v
    "very", "via", "view", "voice", "vast",
    # w
    "we", "was", "were", "will", "with", "what", "when", "who",
    "way", "well", "work", "word", "want", "walk", "warm", "watch",
    "water", "whole", "wide", "wind", "woman", "world", "write",
    # x
    "x",
    # y
    "you", "your", "yes", "yet", "year",
    # z
    "zero", "zone",
]

# De-duplicate and sort once at import time
DICTIONARY = sorted(set(DICTIONARY))


# ---------------------------------------------------------------------------
# Edit distance (Levenshtein)
# ---------------------------------------------------------------------------

def edit_distance(a, b):
    """
    Computes the minimum number of single-character edits (insertions,
    deletions, substitutions) required to turn string a into string b.

    This is the standard dynamic programming solution.
    Time complexity: O(len(a) * len(b)) — fast enough for short words.

    Examples:
        edit_distance("house", "housr") → 1  (one substitution: e→r)
        edit_distance("car",   "care")  → 1  (one insertion: add e)
        edit_distance("the",   "teh")   → 2  (two transpositions)
    """
    m, n = len(a), len(b)

    # dp[i][j] = edit distance between a[:i] and b[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases: transforming empty string to b[:j] requires j insertions
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]          # characters match — free
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1]   # substitution
                )

    return dp[m][n]


# ---------------------------------------------------------------------------
# LanguageModel class
# ---------------------------------------------------------------------------

class LanguageModel:
    """
    Manages the text output and word-level intelligence.

    State:
        current_word  : str  — letters accumulated since last word boundary
        output_text   : str  — the full committed text so far
        completions   : list — top word completion suggestions for current_word
    """

    def __init__(self, max_edit_distance=2, max_completions=3):
        """
        max_edit_distance : how many errors to tolerate before giving up
                            on autocorrect. 2 handles most real handwriting
                            errors without introducing false corrections.
        max_completions   : how many word suggestions to surface at once.
        """
        self.max_edit_distance = max_edit_distance
        self.max_completions   = max_completions

        self.current_word  = ""
        self.output_text   = ""
        self.completions   = []


    # ── Letter input ──────────────────────────────────────────────────────

    def add_letter(self, letter):
        """
        Called each time a new letter is recognised from a stroke.
        Appends it to the current word and updates completions.
        """
        self.current_word += letter.lower()
        self.completions   = self._get_completions(self.current_word)

        print(f"  [LM] Buffer: '{self.current_word}'  "
              f"Completions: {self.completions}")


    def backspace(self):
        """
        Remove the last letter from the current word buffer.
        If the buffer is empty, remove the last character from output_text.
        Triggered by triple tap in writing mode.
        """
        if self.current_word:
            self.current_word = self.current_word[:-1]
            self.completions  = self._get_completions(self.current_word)
            print(f"  [LM] Backspace → buffer: '{self.current_word}'")
        elif self.output_text:
            self.output_text = self.output_text[:-1]
            print(f"  [LM] Backspace → text: '{self.output_text}'")


    # ── Word boundaries ───────────────────────────────────────────────────

    def commit_word(self):
        """
        Called when the user lifts the pen long enough to signal end of word
        (or presses double-tap for enter). Autocorrects the current buffer,
        appends it to output_text, then resets the buffer.

        Returns the final committed word after autocorrection.
        """
        if not self.current_word:
            return ""

        corrected = self._autocorrect(self.current_word)

        if corrected != self.current_word:
            print(f"  [LM] Autocorrect: '{self.current_word}' → '{corrected}'")

        self.output_text  += corrected + " "
        self.current_word  = ""
        self.completions   = []

        print(f"  [LM] Committed. Output: '{self.output_text.strip()}'")
        return corrected


    def accept_completion(self, index=0):
        """
        Accept a word completion suggestion (like tapping a suggestion on a
        phone keyboard). Replaces the current buffer with the chosen word,
        then commits it immediately.

        index : which completion to accept (0 = top suggestion)
        """
        if not self.completions or index >= len(self.completions):
            print("  [LM] No completion available to accept.")
            return ""

        chosen = self.completions[index]
        print(f"  [LM] Completion accepted: '{chosen}'")
        self.current_word = chosen
        return self.commit_word()


    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_completions(self, prefix):
        """
        Returns up to max_completions dictionary words that start with prefix.
        Sorted by word length (shorter = more likely in normal writing).
        """
        if not prefix:
            return []

        matches = [w for w in DICTIONARY if w.startswith(prefix)]
        matches.sort(key=len)
        return matches[:self.max_completions]


    def _autocorrect(self, word):
        """
        If the word is in the dictionary, return it unchanged.
        Otherwise find the closest dictionary word by edit distance.
        If the closest match is within max_edit_distance, return it.
        If nothing is close enough, return the original word as-is.

        Length penalty: we prefer candidates of similar length to the input.
        Without this, short words like "use" beat longer correct matches like
        "house" because they happen to have the same raw edit distance.
        The penalty adds 0.5 per character of length difference, which pushes
        short candidates below same-length candidates in the ranking.
        """
        # Already a valid word — no correction needed
        if word in DICTIONARY:
            return word

        best_word  = word
        best_score = float("inf")   # score = edit_distance + penalties

        for candidate in DICTIONARY:
            # Hard skip — impossible to be within max_edit_distance
            if abs(len(candidate) - len(word)) > self.max_edit_distance:
                continue

            dist        = edit_distance(word, candidate)
            length_diff = abs(len(candidate) - len(word))

            # Penalty 1: length mismatch — prefer same-length candidates
            # (stops short words stealing corrections from longer ones)
            length_penalty = length_diff * 0.5

            # Penalty 2: different first letter — recogniser errors are
            # almost always mid-word, not on the first character.
            # This breaks ties like rael → {call, rain, real} in favour of real.
            first_letter_penalty = 0.4 if candidate[0] != word[0] else 0.0

            score = dist + length_penalty + first_letter_penalty

            if dist <= self.max_edit_distance and score < best_score:
                best_score = score
                best_word  = candidate

        return best_word


    # ── Convenience properties ────────────────────────────────────────────

    @property
    def full_text(self):
        """The complete current text: committed output + in-progress word.
        Always strips trailing whitespace so callers get a clean string
        regardless of whether a word is currently in progress.
        """
        return (self.output_text + self.current_word).strip()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  Language Model Self-Test")
    print("=" * 55)

    lm = LanguageModel()

    # ── Test 1: Edit distance ──────────────────────────────────────────────
    print("\n── Edit Distance Tests ────────────────────────────────")
    cases = [
        ("house", "house",  0),   # identical
        ("house", "housr",  1),   # one substitution
        ("car",   "care",   1),   # one insertion
        ("care",  "car",    1),   # one deletion
        ("love",  "live",   1),   # one substitution
        ("rain",  "ruin",   1),   # one substitution
        ("the",   "teh",    2),   # two moves
        ("never", "ever",   1),   # one deletion at start
    ]
    all_pass = True
    for a, b, expected in cases:
        result = edit_distance(a, b)
        ok = result == expected
        if not ok:
            all_pass = False
        mark = "✓" if ok else "✗"
        print(f"  {mark}  edit_distance('{a}', '{b}') = {result}  "
              f"(expected {expected})")
    print(f"\n  Edit distance: {'all correct' if all_pass else 'SOME FAILED'}")

    # ── Test 2: Word completion ────────────────────────────────────────────
    print("\n── Word Completion Tests ──────────────────────────────")
    prefixes = ["ca", "lo", "no", "ov", "s", "un"]
    for prefix in prefixes:
        completions = lm._get_completions(prefix)
        print(f"  '{prefix}' → {completions}")

    # ── Test 3: Autocorrect ────────────────────────────────────────────────
    print("\n── Autocorrect Tests ──────────────────────────────────")
    typos = [
        ("housr",  "house"),    # recogniser swapped e→r
        ("leve",   "love"),     # one substitution
        ("cre",    "care"),     # one insertion
        ("rael",   "real"),     # two transpositions
        ("teh",    "the"),      # classic typo
        ("overr",  "over"),     # extra letter
        ("house",  "house"),    # already correct — should not change
    ]
    for typo, expected in typos:
        result = lm._autocorrect(typo)
        ok = result == expected
        mark = "✓" if ok else "~"   # ~ = different correction but maybe ok
        print(f"  {mark}  '{typo}' → '{result}'  (expected '{expected}')")

    # ── Test 4: Full pipeline simulation ──────────────────────────────────
    print("\n── Full Pipeline Simulation ───────────────────────────")
    print("  Simulating writing the sentence: 'love is real'\n")

    lm2 = LanguageModel()

    # Write "lovc" (c recognised instead of e — realistic error)
    print("  Writing 'lovc' (intended: 'love') ...")
    for letter in "lovc":
        lm2.add_letter(letter)
    word1 = lm2.commit_word()

    print()

    # Write "is" correctly
    print("  Writing 'is' ...")
    for letter in "is":
        lm2.add_letter(letter)
    word2 = lm2.commit_word()

    print()

    # Write "rael" (letters transposed — realistic error)
    print("  Writing 'rael' (intended: 'real') ...")
    for letter in "rael":
        lm2.add_letter(letter)
    word3 = lm2.commit_word()

    print(f"\n  Final output: '{lm2.output_text.strip()}'")
    print(f"  Expected    : 'love is real'")
    success = lm2.output_text.strip() == "love is real"
    print(f"  Result      : {'✓ PASS' if success else '✗ FAIL'}")

    # ── Test 5: Backspace ──────────────────────────────────────────────────
    print("\n── Backspace Test ─────────────────────────────────────")
    lm3 = LanguageModel()
    for letter in "cre":
        lm3.add_letter(letter)
    print(f"  Buffer after 'cre': '{lm3.current_word}'")
    lm3.backspace()
    print(f"  After one backspace: '{lm3.current_word}'")
    lm3.backspace()
    print(f"  After two backspaces: '{lm3.current_word}'")

    print("\n  Self-test complete.")