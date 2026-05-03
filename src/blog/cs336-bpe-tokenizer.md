---
author: Fred Chyan
pubDatetime: 2026-04-20T10:00:00Z
title: "From Code Points to Subwords: Building a Byte-Level BPE Tokenizer"
slug: cs336-bpe-tokenizer
featured: true
draft: false
tags:
  - machine-learning
  - deep-learning
  - cs336
  - tokenization
  - bpe
description: "A walk through tokenization as taught in Stanford's CS336, from Unicode primitives to a full byte-level BPE tokenizer with encode/decode. Covers why UTF-8 wins, how BPE as compression generalizes to language modeling, and the data structures that make the merge loop tractable on a 10GB corpus."
---

The first thing any language model does is turn text into integers. CS336's opening module spends a whole lecture and most of the first assignment on this step — not because it's glamorous, but because every downstream decision (vocabulary size, context length, multilingual coverage, out-of-vocabulary behavior) gets baked in here. Tokenization is the data contract between your corpus and your model.

This post is a walk through tokenization as taught in Stanford's [CS336](https://cs336.stanford.edu/), from Unicode primitives up through a full byte-level BPE tokenizer with encode/decode. It serves as both my study notes for the course and a practical guide for anyone implementing a tokenizer from scratch. The theory comes from Lecture 1 and the assignment handout; the implementation details are from my own solution to Assignment 1.

## Table of contents

## Part I: The Unicode Foundation

### Code points and the vocabulary problem

Unicode maps characters to integer *code points*. As of Unicode 17.0 (September 2025), the standard defines 159,801 characters across 172 scripts. The character `s` is code point 115 (`U+0073`); the character `牛` is code point 29275.

```python
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```

If we wanted to train a tokenizer whose vocabulary was "one entry per code point," we'd need roughly 160K slots — most of which would be sparse or never seen, and we'd *still* have OOV problems for novel symbols, unusual emoji, or combining-character sequences. So we go one level lower: bytes.

### A small detour: the null character

*Assignment §2.1, `Problem (unicode1): Understanding Unicode` — answered below.*

**a) What Unicode character does `chr(0)` return?**

`chr(0)` returns `'\x00'` — the **null character** (`U+0000`). It's a control character that historically signifies end-of-string in C. Python strings are length-prefixed, so a null byte can sit anywhere inside a string without truncating it.

**b) How does the character's string representation (`__repr__()`) differ from its printed representation?**

The `__repr__()` is the escape literal `'\x00'` — an unambiguous textual representation. `print()`, on the other hand, outputs nothing visible: the null character is a zero-width control code with no glyph, so it produces no ink on the terminal.

**c) What happens when this character occurs in text?**

```python
>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
```

The repr preserves the null byte inline as `\x00`. The printed output *also* contains the null byte — the string is genuinely 21 characters long — but because the null character has no visible glyph, the terminal draws `"this is a test"` and `"string"` with no visible gap, making the output look identical to `"this is a teststring"`. **Printable is not the same as present.** This is worth internalizing before we start treating byte sequences as structured data.

### Unicode encodings: three ways to serialize

*Assignment §2.2, `Problem (unicode2): Unicode Encodings` — answered across this section and the next. Part (a) here; parts (b) and (c) in "Why you can't decode UTF-8 byte by byte" below.*

**a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes rather than UTF-16 or UTF-32?**

Unicode defines three official encodings — UTF-8, UTF-16, UTF-32 — all of which map code points to byte sequences. Comparing them on `"hello! こんにちは!"`:

| Encoding | Byte length |
|----------|-------------|
| UTF-8    | 23          |
| UTF-16   | 28          |
| UTF-32   | 56          |

UTF-8 uses 1–4 bytes per code point, with 1 byte for ASCII. UTF-16 uses 2 or 4 bytes. UTF-32 uses a fixed 4 bytes for *every* code point. Two concrete reasons to prefer UTF-8:

1. **Compactness.** On text that's mostly ASCII (true for most corpora), UTF-8 produces the shortest byte sequences. Shorter sequences mean fewer pairs for BPE to consider during training and shorter token sequences for the model to process at inference.
2. **Ubiquity.** UTF-8 is the encoding of over 98% of the web. Training and deploying on the same encoding avoids mismatch and handles arbitrary Unicode text cleanly.

There's also a structural reason that matters more than it first appears: byte-level tokenization at all is only viable because a byte vocabulary of 256 is small and fixed. UTF-8's variable-width scheme is what makes this work — every code point fits in 1–4 bytes, so those 256 byte values cover all of Unicode with no OOV.

### Why you can't decode UTF-8 byte by byte

Continuing `Problem (unicode2)`, parts (b) and (c).

**b) Why is the following function incorrect? Provide an example input byte string that yields incorrect results.**

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes) -> str:
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
```

The function decodes each byte independently, then concatenates the characters. This is wrong because UTF-8 is **variable-width** — a single code point occupies 1 to 4 bytes, and the decoder needs to see all bytes of a multi-byte sequence together to reconstruct the character.

It happens to work for ASCII, where every code point is exactly one byte. It breaks the moment you feed it anything else. Example: `"こ".encode("utf-8")` produces `b'\xe3\x81\x93'` — three bytes encoding one Japanese character. Running the function on that input:

```python
>>> decode_utf8_bytes_to_str_wrong(b'\xe3\x81\x93')
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe3 in position 0
```

The byte `0xe3` is not itself a valid UTF-8 encoding. Its leading bits (`11100xxx`) mark it as the first byte of a 3-byte sequence, so the decoder refuses to decode it in isolation — it expects two continuation bytes to follow.

**c) Give a two-byte sequence that does not decode to any Unicode character(s).**

`b'\xe3\x81'`. This is the first two bytes of the three-byte UTF-8 encoding of `"こ"` — a valid *prefix* of a multi-byte sequence, but incomplete. The third continuation byte is missing, so these two bytes cannot represent any complete code point, and `bytes.decode("utf-8")` raises on this structurally invalid input.

The takeaway for BPE: we'll operate at the byte level for training, but arbitrary byte sequences are not always valid UTF-8 strings. This reappears in the `decode` step, where we have to handle model-emitted token sequences that might not round-trip cleanly to text.

## Part II: Byte-Level BPE

### The tokenization tradeoff triangle

Every tokenizer navigates three axes:

- **Granularity**: character/byte vs. word vs. subword
- **Vocabulary size**: 256 (raw bytes) ↔ 50K (GPT-2) ↔ 150K+ (full Unicode)
- **Sequence length**: inversely proportional to granularity

Word-level tokenizers (split-on-whitespace, lookup-in-dictionary) are compact but brittle — any novel word (typos, rare proper nouns, compounds, new slang) hits OOV and falls back to an `<UNK>` token, losing information. Character- or byte-level tokenizers have no OOV but produce very long sequences. A 10-word sentence might be 50 tokens at byte level, and self-attention is $O(n^2)$ in sequence length.

**Byte-Pair Encoding (BPE)** is the pragmatic middle. Start at the byte level (vocabulary of 256, no OOV possible), then merge the most frequent adjacent pairs into new tokens. You end up with a vocabulary where common words and subwords are single tokens, while rare strings decompose gracefully down to bytes.

### BPE as compression, repurposed

BPE was originally a data compression algorithm ([Philip Gage, 1994](https://www.derczynski.com/papers/archive/BPE_Gage.pdf)). Sennrich, Haddow, and Birch [repurposed it for neural machine translation in 2016](https://arxiv.org/abs/1508.07909), and it became the standard subword tokenizer for modern LMs starting with GPT-2.

The algorithm, in its byte-level form:

1. Initialize vocabulary with all 256 byte values, plus any special tokens.
2. Count the frequency of every adjacent pair of tokens in the pre-tokenized corpus.
3. Find the most frequent pair `(A, B)`. Add the merged token `AB` to the vocabulary.
4. Replace every occurrence of `(A, B)` in the corpus with `AB`. Update pair counts.
5. Repeat steps 2–4 until you reach the target vocabulary size.

The output is a vocabulary (integer ID → byte string) and an ordered list of merges. Both are needed at inference: the vocabulary for the ID lookup, the merges for encoding new text.

### A worked example

From the handout. Suppose our corpus is:

```
low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
```

Pre-tokenize by whitespace (for simplicity — we'll do better later):

```
{low: 5, lower: 2, widest: 3, newest: 6}
```

Count adjacent pairs across all pre-tokens, weighted by pre-token frequency:

```
(l,o): 7   (o,w): 7   (w,e): 8   (e,r): 2
(w,i): 3   (i,d): 3   (d,e): 3   (e,s): 9
(s,t): 9   (n,e): 6
```

`(e,s)` and `(s,t)` tie at 9. The tiebreak rule is **prefer the lexicographically greater pair**, so we merge `(s,t)` first. New vocabulary entry: `st`. Update:

```
{lo w: 5, lo w e r: 2, w i d e st: 3, n e w e st: 6}
```

Recompute pair counts. Next merge is `(e, st)` with count 9. And so on. After 6 merges the sequence is `['st', 'est', 'ow', 'low', 'west', 'ne']`.

Now `newest` tokenizes as `[ne, west]` — six characters compressed to two tokens. `widest` tokenizes as `[w, i, d, est]` — BPE didn't give it a whole-word token because it's rarer in the corpus. This asymmetry is exactly the point: frequency-proportional compression.

## Part III: Training BPE for Real

*Assignment §2.4–2.5, `Problem (train_bpe): BPE Tokenizer Training` — the training function described in this section is the deliverable.*

The algorithm above is four lines of pseudocode. Making it fast enough to train on a 2GB corpus — let alone an 11GB one — is where the engineering happens.

### Pre-tokenization: regex, then merge within chunks

If you merge bytes directly over a raw corpus, you'll produce tokens that straddle word boundaries. `"dog"` and `"dog."` would have completely different IDs with no shared structure. Semantically they're nearly identical.

The fix is **pre-tokenization**: split the corpus into coarse chunks (words, digit runs, punctuation runs) before counting pairs, and restrict merges so they never cross a chunk boundary.

GPT-2 uses a single regex, [lifted from OpenAI's tiktoken](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py):

```python
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

Reading the alternations:

- `'(?:[sdmt]|ll|ve|re)` — English contractions like `'s`, `'t`, `'ll`, `'ve`, `'re`
- ` ?\p{L}+` — a run of letters, optionally preceded by a space
- ` ?\p{N}+` — same for digits
- ` ?[^\s\p{L}\p{N}]+` — a punctuation/symbol run
- `\s+(?!\S)` — trailing whitespace at end of line
- `\s+` — other whitespace

The `\p{L}` and `\p{N}` classes require the `regex` package; Python's stdlib `re` doesn't support Unicode property escapes. Use `re.finditer` (not `re.findall`) to avoid materializing the full match list in memory.

### Special tokens are hard boundaries

Corpora often contain strings like `<|endoftext|>` that mark document boundaries. Merging across these is nonsense — `"end.<|endoftext|>The"` should never produce a token that spans the two documents.

Before running the GPT-2 regex, split on the set of special tokens:

```python
individual_docs = re.split(special_tokens_pattern(special_tokens), chunk)
for doc in individual_docs:
    for m in re.finditer(PAT, doc):
        ...
```

One implementation detail: when escaping special tokens for the regex, **sort by length descending** so that longer tokens match before their prefixes. Regex alternation is first-match, not longest-match, so if one special token is a prefix of another (`<|eot|>` and `<|eot|><|eot|>`), order matters.

The special tokens themselves are added to the vocabulary as single whole-token entries, so they survive encoding unchanged.

### Memory versus throughput: chunked multiprocessing

Pre-tokenization is embarrassingly parallel — each document is independent. But naive multiprocessing runs into a memory wall on large corpora. If you give each of $N$ workers a chunk of size $S$, peak memory is $N \cdot S$. On an 11GB OpenWebText corpus with 16 processes, that's already over your RAM budget before you start counting pairs.

My approach:

1. Seek chunk boundaries at `<|endoftext|>` marks so we never split a document. The handout provides a `find_chunk_boundaries` helper.
2. Split the file into a large number of small chunks (I use 2000).
3. Process them in **batches** of 25 chunks at a time via a `multiprocessing.Pool`. Peak RAM is bounded by the batch size, not the number of processes or the corpus size.

```python
NUM_CHUNKS_TO_PROCESS_AT_ONCE = 25

def multiproc_pretokenize(input_path, special_tokens):
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 2000, b"<|endoftext|>")
        chunks, pretoken_freq = [], Counter()
        last_end = boundaries[-1]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))
            if len(chunks) == NUM_CHUNKS_TO_PROCESS_AT_ONCE or end == last_end:
                with multiprocessing.Pool() as pool:
                    results = pool.map(
                        partial(pretokenize, special_tokens=special_tokens),
                        chunks,
                    )
                for r in results:
                    pretoken_freq += r
                chunks = []
    return pretoken_freq
```

This is the difference between a job that OOMs and a job that finishes.

### The merge loop: don't recount what you didn't touch

The naive merge loop is $O(\text{merges} \cdot \text{pairs})$ — for every merge, rescan the entire corpus's pair counts. On a 32K vocabulary over a large corpus, that's intractable.

The key observation: **merging pair $(A, B)$ only changes pair counts inside pre-tokens that contain $AB$**. Every other pre-token is untouched, and its contribution to `pair_freq` doesn't change. We just need the right reverse index.

Three data structures:

- `pretoken_freq: Counter[tuple[bytes, ...]]` — how many times each byte-tuple appears
- `pair_freq: Counter[tuple[bytes, bytes]]` — current pair frequencies, weighted by pre-token counts
- `pair_to_pretokens: defaultdict[pair -> set[pretoken]]` — reverse index from pair to the pre-tokens that contain it

To merge $(A, B)$:

1. Find the argmax of `pair_freq` (tiebreak: lexicographically greater).
2. For each pre-token in `pair_to_pretokens[(A, B)]`:
   - Subtract its old pair contributions from `pair_freq`.
   - Apply the merge to get the new pre-token.
   - Add the new pair contributions.
   - Update `pair_to_pretokens` for any new pairs introduced.

```python
class Tracker:
    def __init__(self, pretoken_freq):
        self.pretoken_freq = pretoken_freq
        self.pair_freq = Counter()
        self.pair_to_pretokens = defaultdict(set)
        for pretoken, count in pretoken_freq.items():
            for pair in get_pairs(pretoken):
                self.pair_freq[pair] += count
                self.pair_to_pretokens[pair].add(pretoken)

    def merge(self):
        pair_to_merge = find_most_freq_pair(self.pair_freq)
        for pretoken in self.pair_to_pretokens[pair_to_merge]:
            if pretoken not in self.pretoken_freq:
                continue
            count = self.pretoken_freq[pretoken]
            new_pretoken = merge_pair_in_pretoken(pretoken, pair_to_merge)
            for old_pair in get_pairs(pretoken):
                self.pair_freq[old_pair] -= count
            for new_pair in get_pairs(new_pretoken):
                self.pair_freq[new_pair] += count
                self.pair_to_pretokens[new_pair].add(new_pretoken)
            self.pretoken_freq[new_pretoken] += count
            del self.pretoken_freq[pretoken]
        return pair_to_merge
```

With this structure, each merge step touches only the pre-tokens that are actually affected. TinyStories (~2GB) trains to a 10K vocabulary in under two minutes on a modern laptop.

## Part IV: Experiments

Two corpora, two regimes.

### TinyStories (10K vocab, ~2GB)

*Assignment §2.5, `Problem (train_bpe_tinystories): BPE Training on TinyStories` — answered below.*

**a) Training time and memory, longest token, does it make sense?**

Running `train_bpe` on the TinyStories training set (~2GB) with `vocab_size=10000` and `<|endoftext|>` as the only special token finishes in a few minutes on a modern laptop. Peak memory stays in the low single-digit gigabytes — bounded by the pretokenization batch size and the pair-count dictionaries rather than by the corpus size, since the file is read in chunks and never held in memory all at once.

The longest token in the resulting vocabulary is a common English word with a leading space, for example ` accomplishment`. This makes sense: at 10K vocab on a narrow-distribution, relatively clean corpus (children's stories), BPE has enough vocabulary budget to memorize whole common words as single tokens. The leading space is expected because the GPT-2 pre-tokenizer regex groups a leading space with the following letter run, so `" word"` (rather than `"word"`) is the pre-token BPE sees and compresses.

**b) What part of the tokenizer training process takes the most time?**

On TinyStories at 10K vocab, **pre-tokenization dominates** — it accounts for roughly 90% of wall time. The corpus is large relative to the vocabulary target, so the merge loop has relatively few steps (about 10,000 − 256 − 1 ≈ 9,743 merges). The regex pass over ~2GB of text is the expensive part. On larger corpora with larger vocabularies the balance flips, which we'll see next.

### OpenWebText (32K vocab, ~11GB)

*Assignment §2.5, `Problem (train_bpe_expts_owt): BPE Training on OpenWebText` — answered below.*

**a) Longest token, does it make sense?**

OpenWebText (~11GB) is about 5× the size of TinyStories, and the 32K vocab target means more than 3× as many merges (~31,744 versus ~9,744). This shifts the bottleneck from CPU to memory. On a machine with 16GB of RAM, the path to a successful run was three failed attempts away, and each failure traces to a specific data-structure choice.

*Attempt 1: chunks = processes.* The handout's starter code splits the file into one chunk per process, then dispatches through `multiprocessing.Pool`. On 16GB of RAM this crashed immediately. The arithmetic is brutal: even dividing an 11GB file across $N$ workers, the pool accumulates copies (input lists, return values, intermediate `Counter`s) and total live data easily exceeds $2 \times 11\text{GB} = 22\text{GB}$. Hard OOM before the first pre-tokenization pass finishes.

*Attempt 2: many small chunks, batched processing.* The fix for pre-tokenization — and the one that made it into the final code — is to split the corpus into many small chunks (I use 2000 total) and process them in *batches* of 25 at a time through the pool. Peak live data is now bounded by $25 \times \text{chunk\_size}$, a few hundred MB, independent of the corpus size or the process count. Pre-tokenization finished cleanly. But the merge loop then crashed with a different memory error. Watching `htop` during the failure, the issue was swap: the pair-count `Counter` plus the reverse-index `defaultdict[pair -> set[pretoken]]` grew past physical RAM into a 4GB swap file — not enough headroom.

*Attempt 3: larger swap.* Bumping the swap file to 100GB gave the merge loop enough virtual address space to finish. Peak usage settled around 16GB physical + under 10GB swap, and the full training took about 110 minutes. Swap is slow, but it's slow *and successful* — versus OOMing 8,000 merges into 31,000 with nothing to show for it.

The deeper lesson is that the "obvious" Python data structures don't scale sublinearly with corpus size. `Counter[tuple[bytes, bytes]]` has meaningful per-entry overhead, and `defaultdict[pair -> set[tuple[bytes, ...]]]` stores each pre-token tuple by reference in every pair's set. If I were to redo this on a RAM-constrained machine, the right move would be to compress the reverse index to store *integer handles* into the pre-token table rather than the tuples themselves — that alone would cut its size by roughly the average pre-token length.

With the training finally completed, the longest token in the resulting vocabulary turned out to be a bizarre `ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ...` sequence. It looks like broken output, but it's not a bug. Grep the raw corpus and the string actually appears hundreds of times. It's a **mojibake artifact**: somewhere upstream in the scraping pipeline, Latin-1-encoded bytes got re-encoded as UTF-8, producing a deterministic garbage sequence. Because the pipeline produced it repeatedly across scraped pages, BPE saw it as a frequent substring and reserved a vocabulary slot. So yes, it makes sense — the tokenizer is faithfully compressing what it saw.

**b) Compare and contrast the tokenizer trained on TinyStories vs. OpenWebText.**

The two vocabularies have opposite flavors:

- **TinyStories (10K):** long tokens are whole common English words (` accomplishment`, ` forever`, ` cheese`). The vocabulary reads like a dictionary of everyday English. This follows from the corpus being narrow-distribution, the vocab budget being large relative to the common-word set, and the data being clean.
- **OpenWebText (32K):** long tokens include a mix of legitimate multi-word or rare-word chunks *and* scrape artifacts like `ÃÂÃÂ...`. The wider distribution and larger vocab budget mean BPE spends slots on long, corpus-specific byte sequences — including encoding glitches baked in by the pipeline.

The broader lesson: **your tokenizer is a mirror of your corpus**. Clean data gives you a clean vocabulary. Web scrape gives you weird tokens — which later manifest as weird LM behavior (c.f. the [SolidGoldMagikarp](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation) tokens in GPT-2).

The time profile also flips. On OpenWebText at 32K vocab, pre-tokenization is only ~10% of wall time; the **merge loop dominates** at ~90%. The crossover versus TinyStories is a function of `merges × avg_pair_work` vs. `corpus_size × regex_cost` — OWT has 3× the merges, and each merge does more work because the pair dictionaries are much larger.

The merge loop is single-threaded because the pair-count and reverse-index data structures are mutable shared state: each merge depends on the previous one's update. Parallelizing naively would require either locks or an expensive merge-reduce over independent partial updates.

## Part V: Encode and Decode

*Assignment §2.6, `Problem (tokenizer): Implementing the tokenizer` — the `Tokenizer` class described in this section is the deliverable.*

Training produces a vocabulary and an ordered merge list. Inference uses those to encode new text into IDs and decode IDs back into text.

### Encoding: replay the merge history

Given a string, produce a list of integer IDs:

1. **Split on special tokens**, preserving them as atomic matches.
2. **Pre-tokenize** each split using the same GPT-2 regex as training.
3. **Represent each pre-token as a tuple of single-byte `bytes` objects.**
4. **Apply the merges in training order.** Iterate the merge list; for each merge, find the pre-tokens that contain the pair and combine.
5. **Look up each resulting byte sequence** in the vocabulary to get the integer ID.

One subtlety worth pausing on: step 4 is *not* "greedy longest-match in the vocabulary." It's "replay the merge history in order." BPE encoding is deterministic *because* it replays the same sequence of merge decisions that happened at training time, which is what guarantees consistency between training and inference tokenization.

To make step 4 efficient, I use the same reverse index (`pair_to_words`) as training — at encode time, for each merge in the ordered list, we only have to scan the pre-tokens that currently contain that pair.

Encoding a document from the handout: for input `"the cat ate"` with a vocab that includes merges for `(t, h)`, `(t, he)`, `(_, c)`, and so on, the final ID sequence is `[9, 7, 1, 5, 10, 3]` — one ID per final pre-token piece.

### Decoding: bytes in, UTF-8 out

Decoding is much simpler. Concatenate the byte sequences for each ID and decode as UTF-8:

```python
def decode(self, ids: list[int]) -> str:
    return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
```

The important detail is `errors="replace"`. A user (or, more likely, a language model at inference time) can produce an arbitrary sequence of token IDs. The concatenated byte sequence is not guaranteed to be valid UTF-8 — for example, if the model emits a vocabulary token whose bytes are the *first half* of a multi-byte code point, the decode would fail. With `errors="replace"`, Python substitutes invalid byte runs with `U+FFFD` (the replacement character, `�`) rather than raising.

### Streaming: memory-safe tokenization for big files

For corpora that don't fit in memory:

```python
def encode_iterable(self, iterable):
    for s in iterable:
        yield from self.encode(s)
```

Feed it a file handle and consume the generator lazily. There's a correctness condition here that's easy to miss: the chunks you feed in must align with document boundaries (special tokens), or you'll get different tokenization at the chunk seam than you'd get processing the whole corpus at once. Splitting a line midway through a word, then tokenizing each half independently, produces two incomplete tokenizations instead of one complete one.

## Part VI: Tokenizer Experiments

*Assignment §2.7, `Problem (tokenizer_experiments)` — answered below.*

The tokenizer is built; now measure it. Sample 10 documents from each corpus, encode them with each tokenizer, and report compression in **bytes per token** (higher = better compression). Then estimate end-to-end throughput.

**a) Compression on matched corpus/tokenizer pairs.**

| Corpus | Tokenizer | Bytes/token |
|---|---|---|
| TinyStories | TinyStories (10K) | 4.04 |
| OpenWebText | OpenWebText (32K) | 4.57 |

OWT achieves higher compression because the tokenizer has 3.2× the vocabulary (32K vs 10K) and the corpus has more long, structurally repeated chunks (URLs, common multi-word phrases, web-scrape artifacts) that compress into single tokens. TinyStories caps at ~4 bytes/tok because its vocabulary, by design, mostly resolves to whole words rather than to longer multi-word chunks.

**b) Compression when the tokenizer is from the wrong corpus.**

| Corpus | Tokenizer | Bytes/token |
|---|---|---|
| OpenWebText | TinyStories (10K) | **3.33** |
| OpenWebText | OpenWebText (32K) | 4.57 |

Compression collapses from 4.57 to 3.33 — a ~27% drop. Two reasons stack:

1. **Smaller vocab.** 10K slots can't span the breadth of vocabulary in web scrape, so many web-text n-grams have no learned merge.
2. **Domain mismatch.** Even if vocab budget weren't the bottleneck, TinyStories merges encode "the cat sat on the mat" patterns, not `https://`, `}); `, or `<div`. OWT's frequent substrings simply aren't in the merge list.

When a token isn't covered by long merges, BPE falls back closer to byte level — more tokens per byte, lower compression. The reverse direction is more forgiving: encoding TinyStories with the OWT tokenizer gives 3.91 bytes/tok (vs. 4.04 native) — only a ~3% loss, because the OWT vocabulary is essentially a superset of common English words.

**c) Tokenizer throughput and the Pile estimate.**

Two numbers, depending on whether you allow process-level parallelism.

*Single-thread benchmark*, on the small sample from part (a) (single process, no batching):

```
0.06s elapsed, 0.931 MB/s
```

At that rate, the Pile (825 GiB ≈ 8.86 × 10¹¹ bytes) would take roughly **11 days**:

$$ \frac{825 \times 2^{30} \text{ B}}{0.931 \times 10^6 \text{ B/s}} \cdot \frac{1}{86400 \text{ s/day}} \approx 11 \text{ days} $$

*Realistic multiprocessed run.* Encoding the full OpenWebText training set (11 GB) with `multiprocessing.Pool` defaulted to 8 workers, processing 200 chunks of ~55 MB each, took 558 seconds — **21.35 MB/s end-to-end**. Projecting to the Pile: **~11.5 hours**.

The 23× speedup decomposes into roughly:

- **~8× from cores.** `multiprocessing.Pool` defaults to `os.cpu_count()` workers.
- **~3× from amortizing fixed overhead.** The tokenizer's per-call setup (regex compile, the `_derive_word_to_pretokens` and `_derive_pair_to_words` dictionaries) is roughly constant per `encode()` call. On the part-(a) benchmark that's encoding ~6 KB per call and the setup dominates. On the real run with ~55 MB chunks, the setup is rounding error.

So the realistic answer is **~11.5 hours on a typical 8-core laptop**. The single-thread number is what you'd report if "throughput" strictly means the tokenizer code as written without process-level parallelism, but it's not what the experiment would actually take.

**d) Why uint16 for storing token IDs?**

The vocabulary size is 32,000 < $2^{16} = 65{,}536$, so token IDs fit in 16 bits. The other choices fail or waste:

- **uint8** ($2^8 = 256$) is far too small — it can only address 256 distinct IDs, less than 1% of the OWT vocabulary.
- **uint32** doubles storage with no benefit — the high 16 bits are guaranteed zero for any vocabulary up to 65K.

Concretely: the OWT training set after encoding is 2.7 × 10⁹ tokens × 2 bytes = ~5.4 GB on disk at uint16. At uint32 it would be ~10.9 GB. For a dataset you'll re-stream every training epoch, that 2× matters for both disk and I/O bandwidth.

## Closing: three things to carry forward

1. **Tokenization is a data contract.** The vocabulary is a compressed summary of the training corpus's distribution. Clean data gives you a clean vocabulary; messy data gives you `ÃÂÃÂÃÂÃÂ...` tokens. Every downstream model behavior inherits from this contract.

2. **Sequential algorithms with mutable state don't trivially parallelize.** BPE training is 90% single-threaded because each merge depends on the previous one. The interesting speedups are all in the data structures (reverse indices, incremental updates) rather than in throwing cores at the problem.

3. **The algorithm is four lines; the engineering is the point.** The difference between a naive BPE implementation and one that actually finishes on a 10GB corpus is not the algorithm. It's chunked I/O, boundary-aware multiprocessing, and the pair-to-pretoken reverse index. CS336's first assignment is really a forcing function to confront those details early, before the model architecture work starts.

Next post in this series: the Transformer LM itself — embeddings, RMSNorm, RoPE, causal multi-head attention, SwiGLU MLPs, and how `(batch, sequence, d_model)` tensors flow through the stack. Everything that turns this sequence of integer IDs into a distribution over the next one.

## References

- Philip Gage. *A New Algorithm for Data Compression.* Dr. Dobb's Journal, 1994.
- Rico Sennrich, Barry Haddow, and Alexandra Birch. [*Neural Machine Translation of Rare Words with Subword Units.*](https://arxiv.org/abs/1508.07909) ACL 2016.
- Alec Radford et al. [*Language Models are Unsupervised Multitask Learners.*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) OpenAI, 2019.
- CS336: Language Modeling from Scratch. [Stanford, Spring 2026.](https://cs336.stanford.edu/) Lecture 1, Assignment 1.
