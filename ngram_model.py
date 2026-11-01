import math
from collections import Counter
from typing import List, Tuple
from functools import lru_cache

class NgramModel:
    def __init__(self, order=3, unk_cutoff=1):
        """
        order: 1,2,3
        add_k: add-k smoothing constant (e.g., 0.1 or 1.0)
        unk_cutoff: tokens with count <= unk_cutoff , become <unk>
        """
        assert order in (1,2,3), "order must be 1,2,3"
        self.order = order
        self.unk_cutoff = unk_cutoff

        # vocab: مجموعه‌ی کلمات یکتا
        # unigram: تعداد وقوع هر کلمه
        # total_tokens: تعداد کل توکن‌ها، با تکرار

        self.vocab = set()
        self.unigram = Counter()
        self.bigrams = Counter()
        self.trigrams = Counter()

        self.total_tokens = 0 #total unigram tokens After <unk> mapping (not unique tokens)

        self.continuation_counts = Counter()
        self.context_counts = Counter()
        self.trigram_context_counts = Counter()
        self.total_bigram_types = 0

    def _tokenize(self, sentence: str):
        # Whitespace tokenization
        return sentence.split()

    def build_vocab(self, sentences: List[str]):
        raw_tokens = Counter()
        for s in sentences:
            raw_tokens.update(self._tokenize(s))

        self.vocab = {w for w,c in raw_tokens.items() if c > self.unk_cutoff} #unique tokens
        self.vocab.add("<unk>")
        self.vocab.add("<s>")
        self.vocab.add("</s>")

    def _map_unk(self, w: str):
        return w if w in self.vocab else "<unk>"

    def train(self, sentences: List[str]):
        """
        Estimate n-gram counts from clean training sentences.
        """
        if not self.vocab:
            self.build_vocab(sentences)

        for s in sentences:
            toks = [self._map_unk(w) for w in self._tokenize(s)]

            if self.order == 3:
                toks = ["<s>", "<s>"] + toks + ["</s>"]
            elif self.order == 2:
                toks = ["<s>"] + toks + ["</s>"]
            else:
                toks = toks + ["</s>"]

            # unigram
            for w in toks:
                if w != "<s>":
                    self.unigram[w] += 1
                    self.total_tokens += 1

            # bigram
            for i in range(1, len(toks)):
                self.bigrams[(toks[i-1], toks[i])] += 1

            # trigram
            for i in range(2, len(toks)):
                self.trigrams[(toks[i-2],toks[i-1], toks[i])] += 1

        # AFTER TRAINING for kneser_ney
        self.continuation_counts = Counter()
        self.context_counts = Counter()

        for (w1, w2) in self.bigrams:
            self.continuation_counts[w2] += 1
            self.context_counts[w1] += 1

        self.total_bigram_types = len(self.bigrams)

        self.trigram_context_counts = Counter()

        for (w1, w2, w3) in self.trigrams:
            self.trigram_context_counts[(w1, w2)] += 1

    @property # .V instead of .V()
    def V(self):
        # vocabulary size for smoothing
        # unique tokens because vocab is a set
        return len(self.vocab)

    def kn_unigram(self, w):
        w = self._map_unk(w)
        cont = self.continuation_counts[w]

        if self.total_bigram_types == 0:
            return 1 / self.V

        return cont / self.total_bigram_types

    def logp_unigram(self, w: str):
        p = max(self.kn_unigram(w), 1e-12)
        return math.log(p)

    @lru_cache(maxsize=100000)
    def logp_bigram(self, prev: str, w: str):
        D = 0.75

        prev = self._map_unk(prev)
        w = self._map_unk(w)

        c_bigram = self.bigrams[(prev, w)]
        c_prev = self.unigram[prev]

        if c_prev == 0:
            return self.logp_unigram(w)

        first = max(c_bigram - D, 0) / c_prev

        lambda_w = (D * self.context_counts[prev]) / c_prev

        p_cont = self.kn_unigram(w)

        p = first + lambda_w * p_cont
        p = max(p, 1e-12)

        return math.log(p)

    @lru_cache(maxsize=100000)
    def logp_trigram(self, prev2: str, prev: str, w: str):
        D = 0.75

        prev2 = self._map_unk(prev2)
        prev = self._map_unk(prev)
        w = self._map_unk(w)

        c_tri = self.trigrams[(prev2, prev, w)]
        c_bi = self.bigrams[(prev2, prev)]

        if c_bi == 0:
            return self.logp_bigram(prev, w)

        first = max(c_tri - D, 0) / c_bi

        lambda_h = (D * self.trigram_context_counts[(prev2, prev)]) / c_bi

        p_backoff = math.exp(self.logp_bigram(prev, w))

        p = first + lambda_h * p_backoff
        p = max(p, 1e-12)

        return math.log(p)

    def log_prob(self, context: Tuple[str, ...], word: str):
        """
        Compute log P(word | context)
        context length must be order-1
        """
        if self.order == 1:
            return self.logp_unigram(word)

        elif self.order == 2:
            if len(context) < 1:
                return self.logp_unigram(word)
            return self.logp_bigram(context[-1], word)

        elif self.order == 3:
            if len(context) < 2:
                return self.logp_bigram(context[-1], word)
            return self.logp_trigram(context[-2], context[-1], word)


    def sentence_logp(self, tokens: List[str]):
        toks = [self._map_unk(w) for w in tokens]

        if self.order == 3:
            toks = ["<s>", "<s>"] + toks + ["</s>"]
        elif self.order == 2:
            toks = ["<s>"] + toks + ["</s>"]
        else:
            toks = toks + ["</s>"]

        lp = 0.0
        if self.order == 1:
            for w in toks:
                lp += self.logp_unigram(w)
        elif self.order == 2:
            for i in range(1, len(toks)):
                lp += self.logp_bigram(toks[i - 1], toks[i])
        else:
            for i in range(2, len(toks)):
                lp += self.logp_trigram(toks[i - 2], toks[i - 1], toks[i])

        return lp

    def perplexity(self, sentences: List[str]):
        total_logp = 0.0
        total_tokens = 0

        for s in sentences:
            tokens = self._tokenize(s)

            total_logp += self.sentence_logp(tokens)

            # count predicted tokens (exclude <s>)
            total_tokens += len(tokens) + 1  # + </s>

        avg_logp = total_logp / max(total_tokens, 1)
        return math.exp(-avg_logp)
