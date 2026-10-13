# Compiler eval corpus

Sixteen short sources never physically built, each with a hand-authored ground-truth
spec (`NAME.txt` -> `NAME.json`). Regenerate with `python scripts/make_corpus.py`.
Score a set of compiled specs with `python -m eval.compiler_eval --compiled DIR`.

| ids | form | what they exercise |
| --- | --- | --- |
| j01-j09 | jigsaw sheets (`piece_grid`) | group instructions expanded per piece; positions in words; chains vs border-first; a rotated piece |
| r01-r04 | rulebooks (`tower_lattice`) | the same four rules in different words; rules that differ from classic Jenga; extra terminal conditions |
| l01-l03 | LEGO manuals (`stud_grid`) | hard (support) vs soft (order) dependencies |

Released unconditionally with the code (DR-REL-1).
