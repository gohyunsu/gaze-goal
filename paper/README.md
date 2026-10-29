# ICCE 2027 reading-goal manuscript

`main.tex` is the four-page US-Letter IEEE conference manuscript source.
`main.pdf` is generated locally and intentionally tracked with the final source.

Build from this directory with the cached `bodex` Tectonic installation:

```bash
conda run -n bodex tectonic --keep-logs --keep-intermediates main.tex
```

The scientific content and numerical tables are tied to:

- `../artifacts/metrics/outer_kd_confirmatory.json`
- `../artifacts/manifests/outer_kd_output_audit.json`
- `../docs/05_KD_CONFIRMATORY_REPORT.md`
- `../artifacts/metrics/postconfirmatory_ablation.json`
- `../artifacts/manifests/postconfirmatory_ablation_audit.json`
- `../docs/09_POSTCONFIRMATORY_ABLATION_PROTOCOL.md`
- `../docs/10_POSTCONFIRMATORY_ABLATION_REPORT.md`
- `../artifacts/metrics/postconfirmatory_seed_extension.json`
- `../artifacts/manifests/postconfirmatory_seed_extension_audit.json`
- `../artifacts/metrics/postfeedback_controls.json`
- `../artifacts/metrics/postfeedback_shuffled_kd.json`
- `../artifacts/manifests/cpu_length_distribution.json`
- `../docs/12_POSTCONFIRMATORY_SEED_EXTENSION_PROTOCOL.md`
- `../docs/13_POSTCONFIRMATORY_SEED_EXTENSION_REPORT.md`
- `../docs/17_POSTFEEDBACK_SHUFFLED_KD_PROTOCOL.md`
- `../docs/18_POSTFEEDBACK_SHUFFLED_KD_REPORT.md`
- `../docs/14_CLAIM_EVIDENCE_AND_TERMINOLOGY_AUDIT.md`
- `../artifacts/manifests/figure1_scanpath_provenance.json`

The author block records the author order and affiliations supplied by the
research team. Nam-Joon Kim is marked as the corresponding author. ICCE 2027
accepts initial papers from 2 to 6 pages; this manuscript uses 4 pages including
references. Final spelling, affiliations, submission approval, and policy
compliance must be confirmed by the research team before upload.
