from openstbench import TranslationEvaluator


"""
Translation quality example.

Required evaluation inputs:
- reference: reference translations as list[str], .txt, or .json.
- target_text: system text outputs as list[str], .txt, or .json.

Optional evaluation inputs:
- source: source text, required only when COMET is enabled.
- target_lang: target language code used to choose the BLEU tokenizer.

Configurable evaluator parameters:
- use_bleu: compute sacreBLEU.
- use_chrf: compute chrF++.
- use_comet: compute COMET; requires the comet extra.
- use_bleurt: compute BLEURT; requires bleurt-pytorch.
- comet_model: local path or remote model id.
- bleurt_path: local BLEURT checkpoint path.
- bleurt_model: local path or remote model id for BLEURT loading.
- device: "cuda", "cpu", or another torch device string.

Output metrics:
- sacreBLEU
- chrF++
- COMET
- BLEURT
"""


def main():
    evaluator = TranslationEvaluator(
        use_bleu=True,
        use_chrf=True,
        use_comet=False,
        use_bleurt=False,
        comet_model="./model/Unbabel/wmt22-comet-da",
        bleurt_path="./model/lucadiliello/BLEURT-20",
        bleurt_model=None,
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
