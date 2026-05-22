"""
DuckDB query builder over the partitioned Parquet output.

DuckDB reads all tenant partitions via a glob and uses hive_partitioning to
expose tenant/year/month as virtual columns — letting us push tenant filters
down to the filesystem level before any data is read.
"""

import math
from pathlib import Path
from typing import Literal
import duckdb

OUTPUT_DIR = Path(__file__).parent.parent / "output"
PARQUET_GLOB = str(OUTPUT_DIR / "**" / "*.parquet")

Granularity = Literal["day", "month"]

_TRUNC = {"day": "day", "month": "month"}


def _conn() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


def _clean(records: list[dict]) -> list[dict]:
    """Replace float NaN with None so FastAPI can JSON-serialize the response."""
    return [
        {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}
        for row in records
    ]


def query_usage(
    tenant_ids: list[str],
    start: str,
    end: str,
    granularity: Granularity = "day",
    provider: str | None = None,
    service: str | None = None,
    resource_id: str | None = None,
) -> list[dict]:
    trunc = _TRUNC[granularity]

    tenant_list = ", ".join(f"'{t}'" for t in tenant_ids)

    filters = [
        f"x_TenantId IN ({tenant_list})",
        f"ChargePeriodStart >= '{start}'",
        f"ChargePeriodStart < '{end}'",
    ]
    if provider:
        filters.append(f"Provider = '{provider}'")
    if service:
        filters.append(f"ServiceName = '{service}'")
    if resource_id:
        filters.append(f"ResourceId = '{resource_id}'")

    where = " AND ".join(filters)

    sql = f"""
        SELECT
            DATE_TRUNC('{trunc}', ChargePeriodStart)  AS time_bucket,
            x_TenantId                                AS tenant_id,
            Provider                                  AS provider,
            ServiceName                               AS service_name,
            ServiceCategory                           AS service_category,
            RegionId                                  AS region_id,
            BillingCurrency                           AS currency,
            SUM(BilledCost)                           AS billed_cost,
            SUM(EffectiveCost)                        AS effective_cost,
            SUM(ConsumedQuantity)                     AS consumed_quantity,
            FIRST(ConsumedUnit)                       AS consumed_unit
        FROM read_parquet('{PARQUET_GLOB}', hive_partitioning = true)
        WHERE {where}
        GROUP BY ALL
        ORDER BY time_bucket, tenant_id, provider, service_name
    """

    conn = _conn()
    result = conn.execute(sql).fetchdf()
    result["time_bucket"] = result["time_bucket"].astype(str)
    return _clean(result.to_dict(orient="records"))


def query_anomalies(
    tenant_ids: list[str],
    start: str,
    end: str,
    provider: str | None = None,
    service: str | None = None,
) -> list[dict]:
    """
    Z-score anomaly detection over daily (tenant, provider, service_name) cost.

    Strategy:
      1. Load full history for the tenant(s) to build per-group baselines.
      2. Compute mean + stddev per (tenant, provider, service) over all history.
      3. For rows in [start, end], flag those where |z| > 2.
      4. Severity: low 2–2.5, medium 2.5–3, high >3.

    With only ~30 days of sample data a global mean/std is reasonable.
    In production, use a rolling 30-day window refreshed nightly.
    """
    tenant_list = ", ".join(f"'{t}'" for t in tenant_ids)

    extra_filters = ""
    if provider:
        extra_filters += f" AND Provider = '{provider}'"
    if service:
        extra_filters += f" AND ServiceName = '{service}'"

    sql = f"""
        WITH daily AS (
            SELECT
                DATE_TRUNC('day', ChargePeriodStart)  AS day,
                x_TenantId                            AS tenant_id,
                Provider                              AS provider,
                ServiceName                           AS service_name,
                ServiceCategory                       AS service_category,
                SUM(BilledCost)                       AS daily_cost
            FROM read_parquet('{PARQUET_GLOB}', hive_partitioning = true)
            WHERE x_TenantId IN ({tenant_list})
            {extra_filters}
            GROUP BY ALL
        ),
        baselines AS (
            SELECT
                tenant_id,
                provider,
                service_name,
                AVG(daily_cost)    AS mean_cost,
                STDDEV(daily_cost) AS std_cost
            FROM daily
            GROUP BY tenant_id, provider, service_name
        ),
        scored AS (
            SELECT
                d.*,
                b.mean_cost,
                b.std_cost,
                CASE
                    WHEN b.std_cost = 0 OR b.std_cost IS NULL THEN 0
                    ELSE (d.daily_cost - b.mean_cost) / b.std_cost
                END AS z_score
            FROM daily d
            JOIN baselines b
              ON d.tenant_id   = b.tenant_id
             AND d.provider     = b.provider
             AND d.service_name = b.service_name
        )
        SELECT
            day         AS time_bucket,
            tenant_id,
            provider,
            service_name,
            service_category,
            daily_cost  AS observed_cost,
            mean_cost   AS expected_cost,
            std_cost,
            z_score,
            CASE
                WHEN ABS(z_score) > 3   THEN 'high'
                WHEN ABS(z_score) > 2.5 THEN 'medium'
                ELSE                         'low'
            END AS severity,
            CONCAT(
                ROUND(ABS(z_score), 1), 'σ above' ,
                CASE WHEN z_score > 0 THEN ' above' ELSE ' below' END,
                ' 30-day avg'
            ) AS reason
        FROM scored
        WHERE
            ABS(z_score) > 2
            AND day >= '{start}'
            AND day <  '{end}'
        ORDER BY ABS(z_score) DESC
    """

    conn = _conn()
    result = conn.execute(sql).fetchdf()
    result["time_bucket"] = result["time_bucket"].astype(str)
    return _clean(result.to_dict(orient="records"))
