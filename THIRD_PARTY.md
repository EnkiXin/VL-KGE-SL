# Source provenance and licenses

- The author VL-KGE release is obtained from https://github.com/thefth/vl-kge
  at commit `c78994e14cf2dfda251b701c2803215d9d5fe254`.
  It is not bundled with its datasets or Git history. Its MIT notice is
  preserved in `third_party/vl-kge-LICENSE`; the sole runtime correction is
  recorded in `patches/author-utils-import.patch`.
- `vendor/sl-manifold-core/` is a source snapshot of the shared SL geometry
  implementation used by these experiments. Its numerical limitations and
  implementation provenance are documented in its README and AUDIT files.
  No additional open-source license is assigned to this project-specific code
  by this repository upload.
- No CLIP model weights, dataset feature pickles, training checkpoints, server
  credentials, or machine-specific experiment logs are included.

This is an experimental extension, not an official author implementation of
MuRP or a claim that SL geometry improves every benchmark.
