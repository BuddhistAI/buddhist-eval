# 01: Rename the package to buddhist_eval and set up its foundations

**What to build:** Turn the OpenPecha template into the real `buddhist-eval` package, so that every later ticket can add a scoring function without touching packaging. After this ticket, a developer can install the package and `import buddhist_eval`. It exposes a version and the package's error classes, and has no scoring functions yet. See the RFC at `.scratch/cer-and-span-f1/spec.md` (Implementation Decisions › Package, Errors).

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] The distribution is named `buddhist-eval`, the import package `buddhist_eval`, and the version is `0.1.0`, with `pyproject.toml` as the only source of metadata. Project URLs and the description no longer point at the template.
- [ ] Python 3.10 or later is required, and the mypy setting is raised to match.
- [ ] `jiwer` and `pyewts` are runtime dependencies. Hugging Face `evaluate` is not.
- [ ] The src-layout has `core/` and `utils/` subpackages. `__init__.py` holds only `__version__` and re-exports.
- [ ] There is one package-level base error, with subclasses for empty input, misaligned texts and invalid Spans, all exported from `buddhist_eval`.
- [ ] The template's example module and its test are deleted.
- [ ] A smoke test checks that `buddhist_eval.__version__ == "0.1.0"` and that the error classes are importable and share the base class.
- [ ] CI passes.
