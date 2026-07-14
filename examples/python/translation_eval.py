from openstbench import TranslationEvaluator


"""
Translation quality example.

Required evaluation inputs:
- reference: reference translations as list[str], .txt, or .json.
- target_text: system text outputs as list[str], .txt, or .json.

Optional evaluation inputs:
- source: source text, required for COMET, COMETKiwi, and MetricX_QE.
- target_lang: target language code used to choose the BLEU tokenizer.

Configurable evaluator parameters:
- use_bleu: compute sacreBLEU.
- use_chrf: compute chrF++.
- use_comet: compute COMET; requires the comet extra.
- use_bleurt: compute BLEURT; requires bleurt-pytorch.
- use_metricx: compute MetricX; enabled by default and requires the metricx extra.
- comet_model: local path or remote model id.
- bleurt_path: local BLEURT checkpoint path.
- bleurt_model: local path or remote model id for BLEURT loading.
- metricx_version: "24" or "23"; defaults to "24".
- metricx_model: local path or remote model id for reference-based MetricX.
- metricx_qe_model: local path or remote model id for QE MetricX.
- metricx_tokenizer: local path or remote tokenizer id; defaults to google/mt5-xl.
- device: "cuda", "cpu", or another torch device string.

Output metrics:
- sacreBLEU
- chrF++
- COMET
- BLEURT
- MetricX: reference-based score, lower is better.
- MetricX_QE: reference-free score, lower is better.

MetricX follows the official google-research/metricx README. It only uses text:
source, target_text/hypothesis, and optionally reference. It does not use audio.
""" 


def main():
    evaluator = TranslationEvaluator(
        use_bleu=True,
        use_chrf=True,
        use_comet=False,
        use_bleurt=False,
        use_metricx=True,
        comet_model="./model/Unbabel/wmt22-comet-da",
        bleurt_path="./model/lucadiliello/BLEURT-20",
        bleurt_model=None,
        metricx_version="24",
        metricx_model="google/metricx-24-hybrid-large-v2p6",
        metricx_qe_model=None,
        metricx_tokenizer="google/mt5-xl",
        metricx_batch_size=1,
        device="cuda",
    )

    results = evaluator.evaluate_all(
        reference=["我喜欢看电影。", "今天天气很好。"],
        target_text=["我喜欢看电影。", "今天天气很好。"],
        source=["I like watching movies.", "The weather is nice today."],
        target_lang="zh",
    )

    print(results)


if __name__ == "__main__":
    main()
