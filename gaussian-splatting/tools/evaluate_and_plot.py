import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def ensure_plotting_available():
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting. Install it in the active environment first.")


def run_subprocess(command, workdir):
    print("[RUN]", " ".join(command))
    subprocess.run(command, cwd=workdir, check=True)


def load_json(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path):
    if not path.exists():
        return []

    records = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def flatten_results(model_path, results_data):
    rows = []
    if not results_data:
        return rows

    for method_name, metrics in results_data.items():
        rows.append(
            {
                "model_path": str(model_path),
                "run_label": model_path.name,
                "method": method_name,
                "PSNR": metrics.get("PSNR"),
                "SSIM": metrics.get("SSIM"),
                "LPIPS": metrics.get("LPIPS"),
            }
        )
    return rows


def write_summary_csv(rows, output_path):
    if not rows:
        return

    fieldnames = ["run_label", "model_path", "method", "PSNR", "SSIM", "LPIPS"]
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_training_curves(run_training_data, output_path):
    ensure_plotting_available()

    metric_specs = [
        ("total_loss", "Total Loss"),
        ("l1_loss", "Train L1"),
        ("depth_loss", "Depth Loss"),
        ("test_eval_psnr", "Test PSNR"),
        ("total_points", "Total Points"),
        ("depth_confidence_mean", "Depth Confidence Mean"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()

    for axis, (metric_key, title) in zip(axes, metric_specs):
        has_values = False
        for run_label, records in run_training_data.items():
            x_values = [record["iteration"] for record in records if metric_key in record]
            y_values = [record[metric_key] for record in records if metric_key in record]
            if not x_values:
                continue
            has_values = True
            axis.plot(x_values, y_values, label=run_label, linewidth=1.8)

        axis.set_title(title)
        axis.set_xlabel("Iteration")
        axis.grid(True, alpha=0.3)
        if has_values:
            axis.legend()
        else:
            axis.set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_summary_metrics(summary_rows, output_path):
    ensure_plotting_available()
    if not summary_rows:
        return

    labels = [f"{row['run_label']}\n{row['method']}" for row in summary_rows]
    metrics = [("PSNR", "Higher is better"), ("SSIM", "Higher is better"), ("LPIPS", "Lower is better")]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for axis, (metric_name, subtitle) in zip(axes, metrics):
        values = [row[metric_name] for row in summary_rows]
        axis.bar(range(len(labels)), values)
        axis.set_title(f"{metric_name}\n{subtitle}")
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=35, ha="right")
        axis.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_per_view_metrics(model_path, per_view_data, output_dir):
    ensure_plotting_available()
    if not per_view_data:
        return

    metrics = ("PSNR", "SSIM", "LPIPS")
    for method_name, method_metrics in per_view_data.items():
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        image_names = None

        for axis, metric_name in zip(axes, metrics):
            values_dict = method_metrics.get(metric_name, {})
            current_image_names = sorted(values_dict.keys())
            current_values = [values_dict[name] for name in current_image_names]
            image_names = current_image_names if image_names is None else image_names
            axis.plot(range(len(current_values)), current_values, marker="o", linewidth=1.5)
            axis.set_title(metric_name)
            axis.grid(True, alpha=0.3)

        if image_names:
            tick_step = max(1, len(image_names) // 20)
            tick_positions = list(range(0, len(image_names), tick_step))
            axes[-1].set_xticks(tick_positions)
            axes[-1].set_xticklabels([image_names[index] for index in tick_positions], rotation=45, ha="right")
            axes[-1].set_xlabel("Image")

        fig.suptitle(f"{model_path.name} | {method_name}", y=0.995)
        fig.tight_layout()
        fig.savefig(output_dir / f"per_view_{model_path.name}_{method_name}.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render, evaluate and plot gaussian-splatting runs.")
    parser.add_argument("--model_paths", "-m", nargs="+", required=True, help="Model directories to evaluate.")
    parser.add_argument("--output_dir", type=str, default="", help="Directory where plots and CSV summaries are written.")
    parser.add_argument("--python_exe", type=str, default=sys.executable, help="Python executable used for render.py and metrics.py.")
    parser.add_argument("--iteration", type=int, default=-1, help="Iteration forwarded to render.py when --run_render is enabled.")
    parser.add_argument("--run_render", action="store_true", help="Render train/test images before computing metrics.")
    parser.add_argument("--run_metrics", action="store_true", help="Run metrics.py before plotting.")
    parser.add_argument("--skip_train", action="store_true", help="Forwarded to render.py when --run_render is enabled.")
    parser.add_argument("--skip_test", action="store_true", help="Forwarded to render.py when --run_render is enabled.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    render_script = repo_root / "render.py"
    metrics_script = repo_root / "metrics.py"

    model_paths = [Path(path).resolve() for path in args.model_paths]
    output_dir = Path(args.output_dir).resolve() if args.output_dir else model_paths[0] / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_render:
        for model_path in model_paths:
            command = [args.python_exe, str(render_script), "-m", str(model_path), "--iteration", str(args.iteration)]
            if args.skip_train:
                command.append("--skip_train")
            if args.skip_test:
                command.append("--skip_test")
            run_subprocess(command, repo_root)

    if args.run_metrics:
        command = [args.python_exe, str(metrics_script), "-m"]
        command.extend([str(path) for path in model_paths])
        run_subprocess(command, repo_root)

    run_training_data = {}
    summary_rows = []
    for model_path in model_paths:
        training_records = load_jsonl(model_path / "training_log.jsonl")
        if training_records:
            run_training_data[model_path.name] = training_records

        results_data = load_json(model_path / "results.json")
        summary_rows.extend(flatten_results(model_path, results_data))

        per_view_data = load_json(model_path / "per_view.json")
        if per_view_data:
            plot_per_view_metrics(model_path, per_view_data, output_dir)

    if run_training_data:
        plot_training_curves(run_training_data, output_dir / "training_curves.png")

    if summary_rows:
        write_summary_csv(summary_rows, output_dir / "metrics_summary.csv")
        plot_summary_metrics(summary_rows, output_dir / "summary_metrics.png")

    print(f"[DONE] Evaluation artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
