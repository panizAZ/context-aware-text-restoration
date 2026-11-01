from functools import lru_cache


class Scorer:
    def __init__(self, lm, alpha=1.0, beta=1.0, gamma=-0.2, tau=0.02):
        self.lm = lm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    @lru_cache(maxsize=50000)
    def calculate_levenshtein(self, a, b):
        n, m = len(a), len(b)
        d = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1): d[i][0] = i
        for j in range(m + 1): d[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
                if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                    d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
        return d[n][m]

    @lru_cache(maxsize=50000)
    def calculate_score(self, original, candidate, prev_prev, prev_word, next_word, next_next):
        context_score = 0.0
        # Bigram contextual scorer
        if self.lm.order == 2:
            if prev_word is not None:
                context_score += self.lm.logp_bigram(prev_word, candidate)
            if next_word is not None:
                context_score += self.lm.logp_bigram(candidate, next_word)
        #Trigram contextual scorer
        elif self.lm.order == 3:
            if prev_prev is not None and prev_word is not None:
                context_score += self.lm.logp_trigram(prev_prev, prev_word, candidate)
            if prev_word is not None and next_word is not None:
                context_score += self.lm.logp_trigram(prev_word, candidate, next_word)
            if next_word is not None and next_next is not None:
                context_score += self.lm.logp_trigram(candidate, next_word, next_next)

        unigram_score = self.lm.logp_unigram(candidate)
        edit_penalty = self.calculate_levenshtein(original, candidate)

        return (self.alpha * context_score +
                self.beta * unigram_score +
                self.gamma * edit_penalty)

    def select_best(self, original, candidates,prev_prev, prev_word, next_word, next_next):
        if not candidates:
            return original

        original_score = self.calculate_score(original, original, prev_prev ,prev_word, next_word, next_next)
        best_cand = original
        best_score = original_score

        for c in candidates:
            if c == original: continue
            score = self.calculate_score(original, c, prev_prev ,prev_word, next_word, next_next)
            if score > best_score:
                best_score = score
                best_cand = c

        return best_cand if (best_score - original_score) > self.tau else original

# ---- API wrapper ----

_scorer_cache = {}

def score_candidates(token, candidates, prev_prev, prev_word, next_word, next_next, lm,
                     alpha=1.0, beta=1.0, gamma=-1.0, tau=0.2):
    """
    Required API for assignment.

    Returns the best candidate according to LM + edit distance score.
    """

    key = (id(lm), alpha, beta, gamma, tau)

    if key not in _scorer_cache:
        _scorer_cache[key] = Scorer(lm, alpha, beta, gamma, tau)

    scorer = _scorer_cache[key]

    return scorer.select_best(token, candidates, prev_prev ,prev_word, next_word, next_next)