import pandas as pd

from scripts.evaluate import category_metrics, pair_metrics, predict, prf, split_categories


def test_prf_values():
    m = prf(tp=8, fp=2, fn=4, tn=6)
    assert m["precision"] == 0.8
    assert m["recall"] == round(8 / 12, 4)
    assert m["f1"] == round(2 * 0.8 * (8 / 12) / (0.8 + 8 / 12), 4)
    assert m["accuracy"] == 0.7


def test_prf_handles_zero_denominators():
    m = prf(tp=0, fp=0, fn=0)
    assert m["precision"] is None and m["recall"] is None and m["f1"] is None


def test_split_categories():
    assert split_categories("negation|modality") == {"negation", "modality"}
    assert split_categories("") == set()
    assert split_categories(None) == set()


def _toy_frame():
    return pd.DataFrame([
        {"example_id": "T1", "original_text": "Candidates must submit the form.",
         "rewritten_text": "Candidates may submit the form.", "expected_drift": 1,
         "expected_categories": "modality", "mutation_type": "modality_weakened", "split": "dev"},
        {"example_id": "T2", "original_text": "Pay 75% of the fee.",
         "rewritten_text": "Pay 75% of the fee.", "expected_drift": 0,
         "expected_categories": "", "mutation_type": "unchanged", "split": "dev"},
    ])


def test_pipeline_on_toy_data():
    pred = predict(_toy_frame())
    assert list(pred["predicted_drift"]) == [1, 0]
    assert pair_metrics(pred) == {"tp": 1, "fp": 0, "fn": 0, "tn": 1,
                                  "precision": 1.0, "recall": 1.0, "f1": 1.0, "accuracy": 1.0}
    assert category_metrics(pred)["modality"]["tp"] == 1


def test_benchmark_schema():
    from scripts.evaluate import DATA
    df = pd.read_csv(DATA, keep_default_na=False)
    required = {"example_id", "original_text", "rewritten_text", "expected_drift",
                "expected_categories", "mutation_type", "split"}
    assert required <= set(df.columns)
    assert df["example_id"].is_unique
    assert set(df["split"]) == {"dev", "test"}
    # every base sentence belongs to exactly one split
    assert (df.groupby("base_id")["split"].nunique() == 1).all()
