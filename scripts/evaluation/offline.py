"""Run offline evaluation against the validation or blind test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from galaxy_morphology_classification.evaluation import (
    build_model_comparison,
    evaluate_model,
    extract_split_metrics,
    load_metrics_payload,
    load_model,
    save_confusion_matrix_figure,
)
from galaxy_morphology_classification.training import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SPLIT,
    DEFAULT_VALIDATION_SPLIT,
)
from scripts.evaluation.main import select_model_artifacts


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run offline evaluation for a trained galaxy morphology classifier.",
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="test",
        help="Dataset split used for offline evaluation.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional path to the trained .keras model file.",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=None,
        help="Optional path to the training metrics JSON associated with the model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used for evaluation.",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Optional path to the cleaned dataset CSV.",
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default=None,
        help="Optional path to the directory containing split CSVs.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=DEFAULT_VALIDATION_SPLIT,
        help="Validation split fraction used when regenerating splits.",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=DEFAULT_TEST_SPLIT,
        help="Test split fraction used when regenerating splits.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed used when regenerating splits.",
    )
    parser.add_argument(
        "--compare-with",
        type=str,
        default=None,
        help="Optional metrics JSON used to build a comparison summary against this run.",
    )
    return parser.parse_args()


def resolve_artifacts(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve the target model and training metrics files."""

    if bool(args.model_path) ^ bool(args.metrics_path):
        raise ValueError("--model-path and --metrics-path must be provided together.")

    if args.model_path and args.metrics_path:
        return Path(args.model_path), Path(args.metrics_path)

    model_name, model_path, metrics_path = select_model_artifacts()
    del model_name
    return Path(args.model_path) if args.model_path else model_path, (
        Path(args.metrics_path) if args.metrics_path else metrics_path
    )


def _extract_comparable_payload(payload: dict[str, Any], split_name: str) -> dict[str, Any]:
    """Extract the metrics fragment relevant for model comparison."""

    try:
        result = extract_split_metrics(payload, split_name)
    except KeyError:
        result = extract_split_metrics(payload, "validation")
    result.setdefault("split_name", split_name)
    return result


def main() -> None:
    """Run offline evaluation and persist metrics artifacts."""

    args = parse_args()
    model_path, metrics_path = resolve_artifacts(args)
    training_payload = load_metrics_payload(metrics_path)

    model = load_model(model_path)
    input_scaling = training_payload.get("input_scaling", "zero_one")
    model_name = str(training_payload["model"])

    output_metrics_path = (
        PROJECT_ROOT / "reports" / "metrics" / f"{model_name}_{args.split}_evaluation.json"
    )
    output_figure_path = (
        PROJECT_ROOT
        / "reports"
        / "figures"
        / f"{model_name}_{args.split}_confusion_matrix.png"
    )

    evaluation_payload = evaluate_model(
        model,
        split_name=args.split,
        batch_size=args.batch_size,
        validation_size=args.validation_split,
        test_size=args.test_split,
        random_state=args.random_state,
        csv_path=Path(args.input_csv) if args.input_csv else None,
        splits_dir=Path(args.splits_dir) if args.splits_dir else None,
        input_scaling=input_scaling,
    )
    evaluation_payload["model"] = model_name
    evaluation_payload["model_path"] = str(model_path)
    evaluation_payload["metrics_source"] = str(metrics_path)

    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with output_metrics_path.open("w", encoding="utf-8") as file:
        json.dump(evaluation_payload, file, indent=2, ensure_ascii=False)

    save_confusion_matrix_figure(
        evaluation_payload["confusion_matrix"],
        {
            int(index): class_name
            for index, class_name in evaluation_payload["index_to_class"].items()
        },
        output_figure_path,
        title=f"{model_name} {args.split} confusion matrix",
    )

    comparison_summary = None
    if args.compare_with:
        comparison_payload = load_metrics_payload(Path(args.compare_with))
        comparison_summary = build_model_comparison(
            {
                model_name: _extract_comparable_payload(evaluation_payload, args.split),
                str(comparison_payload["model"]): _extract_comparable_payload(
                    comparison_payload,
                    args.split,
                ),
            }
        )
        comparison_path = PROJECT_ROOT / "reports" / "metrics" / "model_comparison.json"
        with comparison_path.open("w", encoding="utf-8") as file:
            json.dump(comparison_summary, file, indent=2, ensure_ascii=False)

    print(f"Saved evaluation metrics to {output_metrics_path}")
    print(f"Saved confusion matrix figure to {output_figure_path}")
    if comparison_summary is not None:
        print(
            "Updated comparison summary with selected model "
            f"{comparison_summary['selected_model']}"
        )


if __name__ == "__main__":
    main()
