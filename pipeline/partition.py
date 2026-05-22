from pathlib import Path
import pandas as pd


def write_partitioned(df: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """
    Write one Parquet file per (tenant_id, year, month).

    Layout: <output_dir>/tenant=<id>/year=<YYYY>/month=<MM>/data.parquet

    Re-running the same period is idempotent: the file is overwritten.
    Late-arriving data for a prior period only rewrites that tenant+month partition.
    """
    output_dir = Path(output_dir)
    written: list[Path] = []

    if df.empty:
        return written

    df = df.copy()
    df["_year"] = df["ChargePeriodStart"].dt.year.astype(str)
    df["_month"] = df["ChargePeriodStart"].dt.month.astype(str).str.zfill(2)

    for (tenant_id, year, month), group in df.groupby(["x_TenantId", "_year", "_month"]):
        partition_path = output_dir / f"tenant={tenant_id}" / f"year={year}" / f"month={month}"
        partition_path.mkdir(parents=True, exist_ok=True)

        out_file = partition_path / "data.parquet"
        payload = group.drop(columns=["_year", "_month"])
        payload.to_parquet(out_file, index=False, engine="pyarrow")
        written.append(out_file)

    return written
