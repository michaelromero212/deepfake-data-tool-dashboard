"""
pipeline.py — Main pipeline orchestrator.

Ties together all stages:
  1. ingest_data()       — discover files, derive labels
  2. preprocess_file()   — normalize media, extract frames/spectrograms
  3. extract_metadata()  — hash + media metadata
  4. run_detection()     — deepfake detection scoring
  5. validate_dataset()  — schema + quality checks
  6. generate_manifest() — final DatasetManifest + ValidationReport
  7. upload_to_s3()      — optional S3 upload (mock or real)

Run via CLI:
  python -m src.pipeline run --raw-root data/raw --output-dir outputs
  python -m src.pipeline run --upload        # includes mock S3 upload
  python -m src.pipeline run --skip-detection
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from src.detection import run_detection
from src.ingestion import ingest_data
from src.metadata import compute_dataset_version, extract_metadata
from src.preprocessing import preprocess_file
from src.schemas import (
    DatasetManifest,
    Label,
    Sample,
    ValidationStatus,
)
from src.validation import validate_dataset

console = Console()


def generate_manifest(
    raw_root: Path,
    processed_root: Path,
    output_dir: Path,
    skip_detection: bool = False,
    upload: bool = False,
    mock_s3: bool = True,
) -> DatasetManifest:
    """Full pipeline execution. Returns the completed DatasetManifest."""

    console.print(Panel.fit(
        "[bold cyan]deepfake-data-forge[/bold cyan] — Dataset Pipeline",
        subtitle="github.com/you/deepfake-data-forge",
    ))

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Ingestion ────────────────────────────────────────────────
    console.print("\n[bold]Stage 1/5[/bold] Ingesting raw data...")
    discovered = ingest_data(raw_root)

    if not discovered:
        console.print("[red]No files found in raw data root. Exiting.[/red]")
        sys.exit(1)

    # ── Stages 2-4: Preprocess + Metadata + Detection ─────────────────────
    console.print(f"\n[bold]Stages 2-4/5[/bold] Preprocessing, metadata, detection "
                  f"({'skipped' if skip_detection else 'enabled'})...")

    samples: list[Sample] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing files...", total=len(discovered))

        for df in discovered:
            try:
                # Preprocess
                processed = preprocess_file(df.path, df.media_type, processed_root)
                processed_path = processed[0] if isinstance(processed, list) else processed

                # Metadata
                metadata = extract_metadata(df.path, df.media_type)

                # Detection
                detection = None
                if not skip_detection:
                    detection = run_detection(
                        processed_path=processed_path,
                        label=df.label,
                        media_type=df.media_type,
                    )

                sample = Sample(
                    sample_id=df.sample_id,
                    file_path=str(df.path),
                    media_type=df.media_type,
                    label=df.label,
                    label_source=df.label_source,
                    metadata=metadata,
                    detection_result=detection,
                    processed_path=str(processed_path),
                )
                samples.append(sample)

            except Exception as e:
                logger.error(f"Failed to process {df.path.name}: {e}")
                # Don't halt the pipeline — log and continue
            finally:
                progress.advance(task)

    # ── Stage 5: Validation ───────────────────────────────────────────────
    console.print(f"\n[bold]Stage 5/5[/bold] Validating {len(samples)} samples...")
    validated_samples, report = validate_dataset(samples)

    # ── Manifest generation ───────────────────────────────────────────────
    sample_hashes = [s.metadata.sha256_hash for s in validated_samples]
    dataset_version = compute_dataset_version(sample_hashes)

    label_dist = {}
    media_dist = {}
    for s in validated_samples:
        label_dist[s.label.value] = label_dist.get(s.label.value, 0) + 1
        media_dist[s.media_type.value] = media_dist.get(s.media_type.value, 0) + 1

    manifest = DatasetManifest(
        dataset_name="deepfake-data-forge",
        dataset_version=dataset_version,
        created_at=datetime.now(timezone.utc).isoformat(),
        total_samples=len(validated_samples),
        label_distribution=label_dist,
        media_type_distribution=media_dist,
        samples=validated_samples,
    )

    # ── Save outputs ──────────────────────────────────────────────────────
    manifest_path = output_dir / "dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest.model_dump(), f, indent=2, default=str)

    report_path = output_dir / "validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report.model_dump(), f, indent=2)

    # ── Summary table ─────────────────────────────────────────────────────
    _print_summary(manifest, report, manifest_path, report_path)

    # ── Optional S3 upload ────────────────────────────────────────────────
    if upload:
        console.print("\n[bold]Uploading to S3...[/bold]")
        from src.storage import upload_to_s3
        upload_to_s3(manifest_path, processed_root, mock=mock_s3)

    return manifest


def _print_summary(manifest, report, manifest_path, report_path) -> None:
    console.print()

    table = Table(title="Pipeline Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Dataset version", manifest.dataset_version)
    table.add_row("Total samples", str(manifest.total_samples))
    table.add_row("Label distribution", str(manifest.label_distribution))
    table.add_row("Media types", str(manifest.media_type_distribution))
    table.add_row("Validation pass rate", f"{report.pass_rate:.1%}")
    table.add_row("Validation warnings", str(report.warned))
    table.add_row("Validation failures", str(report.failed))
    table.add_row("Manifest path", str(manifest_path))
    table.add_row("Report path", str(report_path))

    console.print(table)

    if report.failed > 0:
        console.print(f"\n[yellow]⚠ {report.failed} samples failed validation. "
                      f"Check validation_report.json for details.[/yellow]")
    else:
        console.print("\n[green]✓ All samples passed validation.[/green]")


# ── CLI ───────────────────────────────────────────────────────────────────

@click.group()
def main():
    """deepfake-data-forge — MLOps dataset pipeline for deepfake detection."""
    pass


@main.command()
@click.option("--raw-root", default="data/raw", show_default=True,
              help="Root directory of raw input media.")
@click.option("--processed-root", default="data/processed", show_default=True,
              help="Output directory for processed media.")
@click.option("--output-dir", default="outputs", show_default=True,
              help="Where to save the manifest and validation report.")
@click.option("--skip-detection", is_flag=True, default=False,
              help="Skip deepfake detection inference stage.")
@click.option("--upload", is_flag=True, default=False,
              help="Upload outputs to S3 after pipeline completes.")
@click.option("--real-s3", is_flag=True, default=False,
              help="Use real AWS S3 instead of moto mock.")
def run(raw_root, processed_root, output_dir, skip_detection, upload, real_s3):
    """Run the full dataset preparation pipeline."""
    generate_manifest(
        raw_root=Path(raw_root),
        processed_root=Path(processed_root),
        output_dir=Path(output_dir),
        skip_detection=skip_detection,
        upload=upload,
        mock_s3=not real_s3,
    )


@main.command()
@click.argument("raw_root")
def stats(raw_root):
    """Quick stats on a raw data directory (no processing)."""
    discovered = ingest_data(Path(raw_root))
    console.print(f"[cyan]{len(discovered)} files discovered[/cyan]")
    for f in discovered[:10]:
        console.print(f"  {f.media_type.value:6s} | {f.label.value:10s} | {f.path.name}")
    if len(discovered) > 10:
        console.print(f"  ... and {len(discovered) - 10} more")


if __name__ == "__main__":
    main()
