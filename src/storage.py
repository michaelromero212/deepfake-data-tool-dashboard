"""
storage.py — Cloud storage stage (AWS S3 / moto mock).

Uploads processed outputs and the dataset manifest to S3.
In real deployments: set AWS credentials and a real bucket name.
In demo/dev mode: moto intercepts all boto3 calls locally — no AWS account needed.

Usage:
  FORGE_S3_MOCK=1 forge run --upload          # local mock
  FORGE_S3_BUCKET=my-bucket forge run --upload # real S3
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import boto3
from loguru import logger

MOCK_BUCKET = "deepfake-data-forge-mock"
MOCK_REGION = "us-east-1"


def _get_client(mock: bool):
    if mock:
        # moto intercepts boto3 — no real AWS calls made
        from moto import mock_aws
        return mock_aws()
    return None


def upload_to_s3(
    manifest_path: Path,
    processed_root: Path,
    bucket_name: str | None = None,
    mock: bool = True,
) -> dict[str, str]:
    """
    Upload the manifest and all processed files to S3.
    Returns a dict mapping local path → S3 URI.

    Args:
        manifest_path: Path to dataset_manifest.json
        processed_root: Root of processed data directory
        bucket_name: S3 bucket name (uses FORGE_S3_BUCKET env var if None)
        mock: If True, use moto mock (no real AWS needed)
    """
    bucket = bucket_name or os.getenv("FORGE_S3_BUCKET", MOCK_BUCKET)
    uploaded: dict[str, str] = {}

    def _do_upload(s3_client) -> None:
        # Ensure bucket exists (mock only — real buckets must exist already)
        if mock:
            s3_client.create_bucket(Bucket=bucket)

        # Upload manifest
        manifest_key = f"manifests/{manifest_path.name}"
        s3_client.upload_file(str(manifest_path), bucket, manifest_key)
        uri = f"s3://{bucket}/{manifest_key}"
        uploaded[str(manifest_path)] = uri
        logger.info(f"Uploaded manifest → {uri}")

        # Upload all processed files
        for file_path in sorted(processed_root.rglob("*")):
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(processed_root.parent)
            key = f"processed/{relative}"
            s3_client.upload_file(str(file_path), bucket, str(key))
            uri = f"s3://{bucket}/{key}"
            uploaded[str(file_path)] = uri

        logger.info(f"Uploaded {len(uploaded)} files to s3://{bucket}/")

    if mock:
        from moto import mock_aws

        @mock_aws
        def _run():
            client = boto3.client("s3", region_name=MOCK_REGION)
            _do_upload(client)

        _run()
        logger.info(
            "[MOCK MODE] S3 upload simulated with moto — no real AWS calls made. "
            "Set FORGE_S3_MOCK=0 and configure AWS credentials for real upload."
        )
    else:
        client = boto3.client("s3")
        _do_upload(client)

    return uploaded
