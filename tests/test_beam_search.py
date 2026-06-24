"""Manual beam search (the tiktoken twin) against scripted fake logits."""

from evals.beam_search import manual_top_k_words

from .fakes import BOS_ID, EOS_ID, FakeBackend


def _run(backend, k=10):
    cont, ws = backend.build_constraint_ids()
    return manual_top_k_words(backend, "ignored prompt", cont, ws, k=k)


def test_manual_beam_produces_expected_words():
    results = _run(FakeBackend())
    words = [w for w, _ in results]
    scores = [s for _, s in results]

    # Scripted paths: "cat", "cats", "dog", "the" are all reachable.
    assert {"cat", "cats", "dog", "the"}.issubset(set(words))
    assert all(w.isalpha() for w in words)            # clean alphabetic words
    assert len(words) == len(set(words))              # deduplicated
    assert "<eos>" not in words and "<bos>" not in words  # no special-token leakage
    assert scores == sorted(scores, reverse=True)     # ranked by descending score


def test_first_token_restricted_to_word_start():
    # BOS favours the bare continuation token "z" (id 9) as the FIRST token, then EOS.
    # "z" is not a space-prefixed word-start token, so the first-token constraint masks
    # it out: no result may begin with "z"; the beam falls back to the (tied) word-start
    # tokens instead.
    transitions = {BOS_ID: {9: 5.0}, 9: {EOS_ID: 5.0}}
    results = _run(FakeBackend(transitions=transitions))
    words = [w for w, _ in results]
    assert "z" not in words
    # Every result begins with a word-start surface (" cat"/" dog"/" the"); none starts
    # with the masked-out continuation token "z".
    assert all(w.startswith(("cat", "dog", "the")) for w in words)


def test_word_start_mass_terminates_the_word():
    # From " cat" the model wants to start a new word (" dog", a word-start token);
    # that mass is redirected to EOS, so the completion is just "cat".
    transitions = {BOS_ID: {0: 5.0}, 0: {1: 5.0}}  # 0=" cat", 1=" dog" (word-start)
    results = _run(FakeBackend(transitions=transitions))
    assert results[0][0] == "cat"
