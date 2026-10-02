# MetaMath scoring

Source: [MetaMath](https://github.com/meta-math/MetaMath/tree/fe667b14f9a4c51bde9809adafb82b6b5d255800),
revision `fe667b14f9a4c51bde9809adafb82b6b5d255800`, Apache-2.0 (see `LICENSE`).

`scoring.py` contains the answer-extraction and scoring functions from
`eval_gsm8k.py` and `eval_math.py`; `util.py` contains their normalization helpers.
Scoring logic is unchanged. Imports use relative `util` and Python's
standard-library `fractions.Fraction`; one string literal uses raw-string syntax
to avoid an invalid-escape warning. Unused functions and command-line code
are omitted. Generation uses Hugging Face Transformers with the MetaMath
prompt, stop strings and token limits; MATH evaluation uses MATH-500.
