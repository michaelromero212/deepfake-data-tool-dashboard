"""
validation.py — Dataset validation stage.

Runs schema checks, file existence checks, corruption checks,
and produces a structured ValidationReport using polars for
aggregation over the full sample set.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from loguru import logger

from src.schemas import (
    Sample,
    ValidationIssue,
    ValidationReport,
    ValidationStatus,
)


def _check_file_exists(sample: Sample) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not Path(sample.file_path).exists():
        issues.append(ValidationIssue(
            issue_type="missing_file",
            message=f"Raw file not found: {sample.file_path}",
            severity=ValidationStatus.FAIL,
        ))
    if sample.processed_path and not Path(sample.processed_path).exists():
        issues.append(ValidationIssue(
            issue_type="missing_processed_file",
            message=f"Processed file not found: {sample.processed_path}",
            severity=ValidationStatus.WARN,
        ))
    return issues


def _check_metadata_integrity(sample: Sample) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if sample.metadata.file_size_bytes == 0:
        issues.append(ValidationIssue(
            issue_type="empty_file",
            message=f"{sample.metadata.file_name} has 0 bytes",
            severity=ValidationStatus.FAIL,
        ))

    if not sample.metadata.sha256_hash or len(sample.metadata.sha256_hash) != 64:
        issues.append(ValidationIssue(
            issue_type="invalid_hash",
            message=f"SHA-256 hash malformed for {sample.metadata.file_name}",
            severity=ValidationStatus.FAIL,
        ))

    return issues


def _check_label_quality(sample: Sample) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    from src.schemas import Label, LabelSource

    if sample.label == Label.UNKNOWN:
        issues.append(ValidationIssue(
            issue_type="unknown_label",
            message=f"Could not derive label for {sample.metadata.file_name}",
            severity=ValidationStatus.WARN,
        ))

    if sample.label_source == LabelSource.INFERRED and sample.detection_result is None:
        issues.append(ValidationIssue(
            issue_type="unverified_inferred_label",
            message=f"Label is inferred but no detection result to back it up",
            severity=ValidationStatus.WARN,
        ))

    return issues


def _check_detection_result(sample: Sample) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if sample.detection_result is None:
        return issues

    score = sample.detection_result.detection_score
    from src.schemas import Label

    # Flag cases where detection score strongly disagrees with label
    if sample.label == Label.REAL and score > 0.8:
        issues.append(ValidationIssue(
            issue_type="label_score_mismatch",
            message=(
                f"Labelled REAL but detection score={score:.2f} "
                f"(high synthetic confidence) — review recommended"
            ),
            severity=ValidationStatus.WARN,
        ))
    elif sample.label == Label.SYNTHETIC and score < 0.2:
        issues.append(ValidationIssue(
            issue_type="label_score_mismatch",
            message=(
                f"Labelled SYNTHETIC but detection score={score:.2f} "
                f"(high real confidence) — review recommended"
            ),
            severity=ValidationStatus.WARN,
        ))

    return issues


def validate_sample(sample: Sample) -> Sample:
    """Run all validation checks on a single sample. Returns updated sample."""
    all_issues: list[ValidationIssue] = []
    all_issues.extend(_check_file_exists(sample))
    all_issues.extend(_check_metadata_integrity(sample))
    all_issues.extend(_check_label_quality(sample))
    all_issues.extend(_check_detection_result(sample))

    if any(i.severity == ValidationStatus.FAIL for i in all_issues):
        status = ValidationStatus.FAIL
    elif any(i.severity == ValidationStatus.WARN for i in all_issues):
        status = ValidationStatus.WARN
    else:
        status = ValidationStatus.PASS

    return sample.model_copy(update={
        "validation_status": status,
        "validation_issues": all_issues,
    })


def validate_dataset(samples: list[Sample]) -> tuple[list[Sample], ValidationReport]:
    """
    Validate all samples. Uses polars for aggregation stats.
    Returns (validated_samples, ValidationReport).
    """
    logger.info(f"Running validation on {len(samples)} samples...")
    validated = [validate_sample(s) for s in samples]

    # Use polars for aggregation — fast even at scale
    rows = [
        {
            "sample_id": s.sample_id,
            "status": s.validation_status.value,
            "issue_count": len(s.validation_issues),
        }
        for s in validated
    ]

    df = pl.DataFrame(rows)
    status_counts = df.group_by("status").agg(pl.count("sample_id").alias("count"))
    counts = {row["status"]: row["count"] for row in status_counts.to_dicts()}

    passed = counts.get("pass", 0)
    warned = counts.get("warn", 0)
    failed = counts.get("fail", 0)
    total = len(validated)

    # Aggregate issue types
    issue_type_counts: dict[str, int] = {}
    failed_ids: list[str] = []
    for s in validated:
        if s.validation_status == ValidationStatus.FAIL:
            failed_ids.append(s.sample_id)
        for issue in s.validation_issues:
            issue_type_counts[issue.issue_type] = issue_type_counts.get(issue.issue_type, 0) + 1

    report = ValidationReport(
        total_samples=total,
        passed=passed,
        warned=warned,
        failed=failed,
        pass_rate=round(passed / total, 4) if total > 0 else 0.0,
        corruption_rate=round(
            issue_type_counts.get("empty_file", 0) / total, 4
        ) if total > 0 else 0.0,
        schema_violation_count=issue_type_counts.get("invalid_hash", 0),
        issues_by_type=issue_type_counts,
        failed_samples=failed_ids,
    )

    logger.info(
        f"Validation done — pass={passed}, warn={warned}, fail={failed} "
        f"(pass rate: {report.pass_rate:.1%})"
    )
    return validated, report
