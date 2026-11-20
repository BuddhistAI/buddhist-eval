# 02: Score CER and WER for one Reference/Prediction pair

**What to build:** A researcher can call `score_cer(reference, prediction, language)` or `score_wer(reference, prediction, language)` and get back an `ErrorRateScore` with the rate and the substitution, deletion, insertion and hit counts, plus the Reference length. For `language="bo"`, both texts are converted to Wylie before scoring (ADR-0001), so Tibetan WER counts syllables. Any other ISO 639-1 code scores the text unchanged. CER and WER are independent functions, and neither computes the other. See the RFC (Implementation Decisions › Text scoring module).

**Blocked by:** 01 (package foundation).

**Status:** claimed

- [ ] `score_cer` and `score_wer` are exported from `buddhist_eval` and take a required `language`.
- [ ] Text preparation strips leading and trailing whitespace, then converts to Wylie with `pyewts` only for `"bo"`.
- [ ] Both return an `ErrorRateScore`, a frozen dataclass shared by CER and WER, with the unrounded rate, substitutions, deletions, insertions, hits and Reference length (characters for CER, words for WER).
- [ ] An empty Prediction, after stripping, scores a rate of 1.0, with every Reference unit counted as a deletion.
- [ ] An empty Reference, after stripping, raises the package's empty-input error.
- [ ] Parity test: on fixed Tibetan pairs, the rates equal `evaluate`'s `cer` and `wer` on `pyewts` Wylie. The expected values are computed once and stored as fixtures, so `evaluate` is not a test dependency.
- [ ] Behaviour tests: a non-Tibetan language passes through unchanged, a single wrong syllable gives the expected edit counts, and the empty-Prediction and empty-Reference cases behave as above.
