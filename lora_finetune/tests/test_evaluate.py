from src.evaluate import compute_metrics, print_report


def _make_results(pairs):
    return [{"true": {"score": t[0], "label": t[1]}, "pred": {"score": p[0], "label": p[1]}} for t, p in pairs]


def test_perfect_accuracy():
    results = _make_results([
        ((1, "nitpick"), (1, "nitpick")),
        ((5, "blocking"), (5, "blocking")),
        ((3, "suggestion"), (3, "suggestion")),
    ])
    metrics = compute_metrics(results)
    assert metrics["accuracy"] == 100.0


def test_zero_accuracy():
    results = _make_results([
        ((1, "nitpick"), (5, "blocking")),
        ((5, "blocking"), (1, "nitpick")),
    ])
    metrics = compute_metrics(results)
    assert metrics["accuracy"] == 0.0


def test_partial_accuracy():
    results = _make_results([
        ((1, "nitpick"), (1, "nitpick")),   # correct
        ((5, "blocking"), (1, "nitpick")),  # wrong
    ])
    metrics = compute_metrics(results)
    assert metrics["accuracy"] == 50.0


def test_none_pred_does_not_crash():
    results = [
        {"true": {"score": 1, "label": "nitpick"}, "pred": None},
        {"true": {"score": 5, "label": "blocking"}, "pred": {"score": 5, "label": "blocking"}},
    ]
    metrics = compute_metrics(results)
    assert metrics["accuracy"] == 50.0


def test_print_report_does_not_crash(capsys):
    results = _make_results([
        ((1, "nitpick"), (1, "nitpick")),
        ((5, "blocking"), (4, "warning")),
    ])
    metrics = compute_metrics(results)
    print_report("TEST MODEL", metrics)
    captured = capsys.readouterr()
    assert "TEST MODEL" in captured.out
    assert "50.0%" in captured.out


def test_empty_results():
    metrics = compute_metrics([])
    assert metrics["accuracy"] == 0.0
