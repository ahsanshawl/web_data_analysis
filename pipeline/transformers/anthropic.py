"""
Anthropic transformer — follows config/anthropic_cost_report.csv mapping rules.

Primary source:  cost_report.parquet       (daily, one row per token_type)
Secondary source: usage_report_messages.parquet (hourly, token counts by api_key)

Strategy:
  - Keep cost_report rows at token_type granularity — each is a FOCUS charge line.
  - Aggregate usage_report to daily grain (floor bucket_start, sum over api_key_id).
  - Pivot usage_report token columns → (token_type, count) pairs.
  - Left-join onto cost_report so each line item carries its actual token count.
  - BilledCost = amount (USD) from cost_report.
  - ConsumedQuantity = token count from usage_report; ConsumedUnit = token_type string.
"""

import json
import pandas as pd

from .base import BaseTransformer

# Maps cost_report.token_type values → usage_report column names
_TOKEN_TYPE_TO_COL = {
    "uncached_input_tokens": "uncached_input_tokens",
    "cache_read_input_tokens": "cache_read_input_tokens",
    "cache_creation_ephemeral_5m_input_tokens": "cache_creation_ephemeral_5m_input_tokens",
    "cache_creation_ephemeral_1h_input_tokens": "cache_creation_ephemeral_1h_input_tokens",
    "output_tokens": "output_tokens",
}

_JOIN_KEYS = ["organization_id", "workspace_id", "model", "service_tier", "context_window"]


def _build_usage_lookup(usage_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly usage_report to daily, then melt into
    (organization_id, workspace_id, model, service_tier, context_window, bucket_start, token_type, token_count)
    so it can be joined onto cost_report rows.
    """
    df = usage_df.copy()
    df["bucket_start"] = df["bucket_start"].dt.floor("D")

    token_cols = list(_TOKEN_TYPE_TO_COL.values())
    agg_cols = _JOIN_KEYS + ["bucket_start"] + token_cols
    # Sum over api_key_id within the day
    daily = df[agg_cols].groupby(_JOIN_KEYS + ["bucket_start"], as_index=False).sum()

    # Melt wide → long so each row is one (token_type, count)
    melted = daily.melt(
        id_vars=_JOIN_KEYS + ["bucket_start"],
        value_vars=token_cols,
        var_name="token_col",
        value_name="token_count",
    )

    # Reverse-map column name → token_type value used in cost_report
    col_to_type = {v: k for k, v in _TOKEN_TYPE_TO_COL.items()}
    melted["token_type"] = melted["token_col"].map(col_to_type)
    return melted.drop(columns=["token_col"])


class AnthropicTransformer(BaseTransformer):
    def __init__(self, usage_df: pd.DataFrame | None = None):
        self._usage_df = usage_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cost = df.copy()
        cost["bucket_start"] = pd.to_datetime(cost["bucket_start"], utc=True)
        cost["bucket_end"] = pd.to_datetime(cost["bucket_end"], utc=True)

        # Join token counts from usage_report
        if self._usage_df is not None:
            usage_lookup = _build_usage_lookup(self._usage_df)
            usage_lookup["bucket_start"] = pd.to_datetime(usage_lookup["bucket_start"], utc=True)
            cost = cost.merge(
                usage_lookup,
                left_on=_JOIN_KEYS + ["bucket_start", "token_type"],
                right_on=_JOIN_KEYS + ["bucket_start", "token_type"],
                how="left",
            )
        else:
            cost["token_count"] = None

        out = pd.DataFrame()

        # Account hierarchy
        out["BillingAccountId"] = cost["organization_id"].astype(str)
        out["SubAccountId"] = cost["workspace_id"].astype(str)

        # Provider chain (all static)
        out["Provider"] = "Anthropic"
        out["Publisher"] = "Anthropic"
        out["InvoiceIssuer"] = "Anthropic"

        # Currency & billing period
        out["BillingCurrency"] = cost["currency"]
        month_start = cost["bucket_start"].dt.tz_localize(None).dt.to_period("M").dt.start_time
        out["BillingPeriodStart"] = month_start.dt.tz_localize("UTC")
        month_end = cost["bucket_start"].dt.tz_localize(None).dt.to_period("M").dt.end_time
        out["BillingPeriodEnd"] = month_end.dt.tz_localize("UTC")

        # Charge period — daily bucket from cost_report
        out["ChargePeriodStart"] = cost["bucket_start"]
        out["ChargePeriodEnd"] = cost["bucket_end"]

        # Charge classification
        out["ChargeCategory"] = "Usage"
        out["ChargeFrequency"] = "Usage-Based"
        out["ChargeDescription"] = cost["description"]

        # Service
        out["ServiceName"] = cost["model"]
        out["ServiceCategory"] = "AI and Machine Learning"

        # Cost — USD from cost_report.amount
        out["BilledCost"] = cost["amount"]
        out["EffectiveCost"] = cost["amount"]

        # Usage — token count from usage_report; unit is the token_type label
        out["ConsumedQuantity"] = cost["token_count"]
        out["ConsumedUnit"] = cost["token_type"]

        # All other FOCUS fields not defined for Anthropic stay None (handled by _ensure_schema)

        out["x_SourceGranularity"] = "daily"
        out["x_Metadata"] = cost.apply(self._build_metadata, axis=1)

        return self._ensure_schema(out)

    def _build_metadata(self, row) -> str:
        return json.dumps({
            "anthropic_model": str(row.get("model", "")),
            "anthropic_service_tier": str(row.get("service_tier", "")),
            "anthropic_context_window": str(row.get("context_window", "")),
            "anthropic_token_type": str(row.get("token_type", "")),
        })
