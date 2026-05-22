import json
import pandas as pd

from .base import BaseTransformer

_TOKEN_COLS = [
    "uncached_input_tokens",
    "cache_creation_ephemeral_5m_input_tokens",
    "cache_creation_ephemeral_1h_input_tokens",
    "cache_read_input_tokens",
    "output_tokens",
]

_GROUP_KEYS = ["organization_id", "bucket_start", "bucket_end", "workspace_id", "model", "service_tier", "context_window"]


class AnthropicTransformer(BaseTransformer):
    """
    Primary source: cost_report.parquet (daily cost by token_type).
    Enriched with: usage_report_messages.parquet (hourly token counts, aggregated to day).
    """

    def __init__(self, usage_df: pd.DataFrame | None = None):
        self._usage_df = usage_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Aggregate cost_report rows: multiple token_type rows per (day, workspace, model) → total cost
        cost = (
            df.groupby(_GROUP_KEYS, as_index=False)["amount"]
            .sum()
            .rename(columns={"amount": "total_cost"})
        )
        cost["bucket_start"] = pd.to_datetime(cost["bucket_start"], utc=True)
        cost["bucket_end"] = pd.to_datetime(cost["bucket_end"], utc=True)

        # Optionally enrich with total token counts from usage_report
        if self._usage_df is not None:
            usage = self._usage_df.copy()
            usage["bucket_start"] = pd.to_datetime(usage["bucket_start"], utc=True).dt.floor("D")
            usage["bucket_end"] = usage["bucket_start"] + pd.Timedelta(days=1)
            for col in _TOKEN_COLS:
                if col not in usage.columns:
                    usage[col] = 0
            usage["total_tokens"] = usage[_TOKEN_COLS].sum(axis=1)
            token_agg = (
                usage.groupby(_GROUP_KEYS, as_index=False)["total_tokens"].sum()
            )
            token_agg["bucket_start"] = pd.to_datetime(token_agg["bucket_start"], utc=True)
            cost = cost.merge(token_agg, on=_GROUP_KEYS, how="left")
        else:
            cost["total_tokens"] = None

        out = pd.DataFrame()
        out["BillingAccountId"] = cost["organization_id"].astype(str)
        out["SubAccountId"] = cost["workspace_id"].astype(str)

        out["Provider"] = "Anthropic"
        out["Publisher"] = "Anthropic"
        out["InvoiceIssuer"] = "Anthropic"

        out["BillingCurrency"] = "USD"
        month_start = cost["bucket_start"].dt.tz_localize(None).dt.to_period("M").dt.start_time
        out["BillingPeriodStart"] = month_start.dt.tz_localize("UTC")
        month_end = cost["bucket_start"].dt.tz_localize(None).dt.to_period("M").dt.end_time
        out["BillingPeriodEnd"] = month_end.dt.tz_localize("UTC")

        out["ChargePeriodStart"] = cost["bucket_start"]
        out["ChargePeriodEnd"] = cost["bucket_end"]

        out["ChargeCategory"] = "Usage"
        out["ChargeFrequency"] = "Usage-Based"

        out["ServiceName"] = cost["model"]
        out["ServiceCategory"] = "AI and Machine Learning"

        out["BilledCost"] = cost["total_cost"]
        out["EffectiveCost"] = cost["total_cost"]

        out["ConsumedQuantity"] = cost["total_tokens"]
        out["ConsumedUnit"] = "tokens"

        # No resource_id concept — api_key not available in cost_report
        out["ResourceId"] = None

        out["x_SourceGranularity"] = "daily"
        out["x_Metadata"] = cost.apply(self._build_metadata, axis=1)

        return self._ensure_schema(out)

    def _build_metadata(self, row) -> str:
        return json.dumps({
            "anthropic_model": str(row.get("model", "")),
            "anthropic_service_tier": str(row.get("service_tier", "")),
            "anthropic_context_window": str(row.get("context_window", "")),
        })
