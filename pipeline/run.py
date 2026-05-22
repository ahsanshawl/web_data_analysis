"""
Entry point: ingest all sources → normalize → partition by tenant.

Usage:
    python -m pipeline.run [--sample-data <dir>] [--output <dir>] [--config <dir>]
"""
import argparse
from pathlib import Path
import pandas as pd

from .transformers.aws import AWSTransformer
from .transformers.gcp import GCPTransformer
from .transformers.azure import AzureTransformer
from .transformers.anthropic import AnthropicTransformer
from .normalize import enrich_tenant, aggregate_daily, union_all
from .partition import write_partitioned


def _load_category_map(path: Path) -> dict:
    df = pd.read_csv(path)
    return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))


def run(sample_data_dir: Path, output_dir: Path, config_dir: Path):
    tenant_mapping = pd.read_csv(config_dir / "tenant_mapping.csv")
    cat_dir = config_dir / "category_mapping"

    aws_cats = _load_category_map(cat_dir / "aws.csv")
    gcp_cats = _load_category_map(cat_dir / "gcp.csv")
    azure_cats = _load_category_map(cat_dir / "azure.csv")

    frames = []

    # --- AWS ---
    aws_path = sample_data_dir / "aws" / "cost_and_usage_report.parquet"
    if aws_path.exists():
        print(f"[AWS] Loading {aws_path}")
        aws_df = pd.read_parquet(aws_path)
        frames.append(AWSTransformer(aws_cats).transform(aws_df))
        print(f"[AWS] {len(aws_df):,} rows transformed")

    # --- GCP ---
    gcp_path = sample_data_dir / "gcp" / "cloud_billing_export.parquet"
    if gcp_path.exists():
        print(f"[GCP] Loading {gcp_path}")
        gcp_df = pd.read_parquet(gcp_path)
        frames.append(GCPTransformer(gcp_cats).transform(gcp_df))
        print(f"[GCP] {len(gcp_df):,} rows transformed")

    # --- Azure ---
    azure_path = sample_data_dir / "azure" / "amortized_cost_export.parquet"
    if azure_path.exists():
        print(f"[Azure] Loading {azure_path}")
        azure_df = pd.read_parquet(azure_path)
        frames.append(AzureTransformer(azure_cats).transform(azure_df))
        print(f"[Azure] {len(azure_df):,} rows transformed")

    # --- Anthropic ---
    cost_path = sample_data_dir / "anthropic" / "cost_report.parquet"
    usage_path = sample_data_dir / "anthropic" / "usage_report_messages.parquet"
    if cost_path.exists():
        print(f"[Anthropic] Loading {cost_path}")
        cost_df = pd.read_parquet(cost_path)
        usage_df = pd.read_parquet(usage_path) if usage_path.exists() else None
        frames.append(AnthropicTransformer(usage_df).transform(cost_df))
        print(f"[Anthropic] {len(cost_df):,} cost rows transformed")

    if not frames:
        print("No source files found — exiting.")
        return

    # --- Normalize ---
    print("\n[Normalize] Unioning all sources...")
    unified = union_all(frames)
    print(f"[Normalize] {len(unified):,} rows before tenant enrichment")

    unified = enrich_tenant(unified, tenant_mapping)
    unmapped = unified["x_TenantId"].isna().sum()
    if unmapped:
        print(f"[Normalize] WARNING: {unmapped} rows have no tenant mapping")

    print("[Normalize] Aggregating to daily grain...")
    unified = aggregate_daily(unified)
    print(f"[Normalize] {len(unified):,} rows after daily aggregation")

    # --- Partition ---
    print(f"\n[Partition] Writing to {output_dir}")
    written = write_partitioned(unified, output_dir)
    print(f"[Partition] Wrote {len(written)} partition files:")
    for p in sorted(written):
        print(f"  {p}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-data", default="sample-data", type=Path)
    parser.add_argument("--output", default="output", type=Path)
    parser.add_argument("--config", default="config", type=Path)
    args = parser.parse_args()
    run(args.sample_data, args.output, args.config)


if __name__ == "__main__":
    main()
