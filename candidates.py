import string
from functools import lru_cache


class CandidateGenerator:
    def __init__(self, vocab):
        self.vocab = set(vocab)
        self.letters = string.ascii_lowercase

    @lru_cache(maxsize=10000)
    def edit_distance_1(self, word):
        if not word:
            return ()

        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        inserts = [L + c + R for L, R in splits for c in self.letters]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        substitutes = [L + c + R[1:] for L, R in splits if R for c in self.letters]

        return tuple(set(deletes + inserts + transposes + substitutes))

    @lru_cache(maxsize=5000)
    def get_candidates(self, word, distance=2):
        if not word:
            return ()

        # اگر کلمه فقط علامت نگارشی است (مثل "." یا "...")
        if all(c in string.punctuation for c in word):
            return (word,)  # فقط خودش رو برگردون

        # short token policy (skip very short tokens)
        #if len(word) <= 2:
            #return (word,)

        # If word in vocabulary (dictionary filtering)
        if word in self.vocab:
            return (word,)

        valid_candidates = set()

        # بررسی تغییرات ساده کیس (Lower/Upper/Capitalize)
        for variant in {word.lower(), word.upper(), word.capitalize()}:
            if variant in self.vocab:
                valid_candidates.add(variant)

        # کاندیداهای فاصله ۱
        d1_candidates = self.edit_distance_1(word)
        valid_candidates.update(w for w in d1_candidates if w in self.vocab)

        # کاندیداهای فاصله ۲ (فقط اگر کاندیدای فاصله ۱ پیدا نشد و طول کلمه مناسب بود)
        if distance == 2 and not valid_candidates and len(word) >= 3:
            for e1 in d1_candidates:
                for e2 in self.edit_distance_1(e1):
                    if e2 in self.vocab:
                        valid_candidates.add(e2)
                        if len(valid_candidates) >= 100:  # محدودیت برای سرعت
                            break
                if len(valid_candidates) >= 100:
                    break

        return tuple(sorted(valid_candidates)[:100])


# ---- API wrapper ----

_generator_cache = {}

def generate_candidates(token, vocab):
    """
    API required by assignment:
    generate_candidates(token, vocab)

    Uses cached CandidateGenerator internally.
    """
    key = id(vocab)

    if key not in _generator_cache:
        _generator_cache[key] = CandidateGenerator(vocab)

    generator = _generator_cache[key]
    return generator.get_candidates(token)
