#!/usr/bin/env python3
"""
Generate LaTeX tables from main evaluation results.
"""

import json
from pathlib import Path
from typing import Dict, Any


def load_main_results() -> Dict[str, Any]:
    """Load the main evaluation results file."""
    main_results_file = "results/main_evaluation_results.json"

    if not Path(main_results_file).exists():
        print(f"Main results file not found: {main_results_file}")
        return {"evaluations": []}

    with open(main_results_file, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_model_name(model_name: str) -> str:
    """Extract a clean model name for display."""
    # Remove common prefixes and clean up
    model_name = model_name.replace("meta-llama/", "meta-llama_")
    model_name = model_name.replace("openai/", "openai_")
    model_name = model_name.replace("Qwen/", "Qwen_")

    # Clean up special characters for LaTeX
    model_name = model_name.replace("-", "_")
    model_name = model_name.replace(".", "_")

    return model_name


def format_accuracy(accuracy: float) -> str:
    """Format accuracy as a decimal string."""
    return f"{accuracy:.2f}"


def get_parse_errors(summary: Dict[str, Any]) -> str:
    """Get parse errors count from summary."""
    errors = summary.get("errors", 0)
    unknown = summary.get("unknown", 0)
    total_errors = errors + unknown

    if total_errors == 0:
        return "--"
    else:
        return str(total_errors)


def generate_latex_table(results: Dict[str, Any]) -> str:
    """Generate LaTeX table from evaluation results."""

    evaluations = results.get("evaluations", [])
    if not evaluations:
        return "% No evaluation results found"

    # Group evaluations by model
    model_groups = {}
    for eval_entry in evaluations:
        summary = eval_entry["summary"]
        model_name = summary.get("model_name", "Unknown")
        clean_model_name = extract_model_name(model_name)

        if clean_model_name not in model_groups:
            model_groups[clean_model_name] = []

        model_groups[clean_model_name].append(summary)

    # Generate LaTeX table
    latex_lines = []
    latex_lines.append(r"\begin{table}[h]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\begin{tabular}{@{}l l l r r r r r@{}}")
    latex_lines.append(r"\toprule")
    latex_lines.append(
        r"\textbf{Model} & \textbf{Task} & \textbf{Language} & \textbf{Total Qs} & \textbf{Correct} & \textbf{Accuracy} & \textbf{Parse Errors} \\"
    )
    latex_lines.append(r"\midrule")

    # Sort models for consistent output
    sorted_models = sorted(model_groups.keys())

    for i, model_name in enumerate(sorted_models):
        summaries = model_groups[model_name]

        # Sort summaries by task type and language
        task_order = {"validity": 0, "symbolization": 1, "countermodel": 2}
        language_order = {"carroll": 0, "english": 1, "default": 2}

        summaries.sort(
            key=lambda s: (
                task_order.get(s.get("question_type", ""), 99),
                language_order.get(s.get("language", ""), 99),
            )
        )

        # Determine number of rows for this model
        num_rows = len(summaries)

        for j, summary in enumerate(summaries):
            task = summary.get("question_type", "Unknown").title()
            language = summary.get("language", "Unknown").title()
            if language == "Default":
                language = "--"
            elif language == "Carroll":
                language = "Carrol"  # Match your example

            total_qs = summary.get("total_questions", 0)
            correct = summary.get("correct_answers", 0)
            accuracy = format_accuracy(summary.get("accuracy", 0))
            parse_errors = get_parse_errors(summary)

            # First row of model group
            if j == 0:
                latex_lines.append(f"\\multirow{{{num_rows}}}{{*}}{{\\parbox[c]{{2.8cm}}{{\\centering {model_name}}}}}")
                latex_lines.append(
                    f"  & {task:<12} & {language:<8} & {total_qs:>3} & {correct:>7} & {accuracy:>6} & {parse_errors:>12} \\\\"
                )
            else:
                latex_lines.append(
                    f"  & {task:<12} & {language:<8} & {total_qs:>3} & {correct:>7} & {accuracy:>6} & {parse_errors:>12} \\\\"
                )

        # Add midrule between models (except after the last one)
        if i < len(sorted_models) - 1:
            latex_lines.append(r"\midrule")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")

    return "\n".join(latex_lines)


def main():
    """Generate LaTeX table from main results."""
    print("Loading main evaluation results...")
    results = load_main_results()

    if not results.get("evaluations"):
        print("No evaluation results found!")
        return

    print(f"Found {len(results['evaluations'])} evaluation results")

    # Generate LaTeX table
    latex_table = generate_latex_table(results)

    # Save to file
    output_file = "results/evaluation_table.tex"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(latex_table)

    print(f"LaTeX table saved to: {output_file}")
    print("\nGenerated table:")
    print("=" * 80)
    print(latex_table)
    print("=" * 80)


if __name__ == "__main__":
    main()
