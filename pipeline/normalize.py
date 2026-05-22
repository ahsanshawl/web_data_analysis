import pandas as pd

from .transformers.base import FOCUS_COLUMNS, MEASURE_COLS

# Dimensions used as the groupby key when aggregating hourly → daily.
# x_Metadata is excluded from the key: we take first value per group to avoid
# over-splitting on tag differences within the same day/resource.
_DAILY_KEY_COLS = [
    "BillingAccountId", "BillingAccountName",
    "SubAccountId", "SubAccountName",
    "Provider", "Publisher", "InvoiceIssuer",
    "BillingCurrency", "BillingPeriodStart", "BillingPeriodEnd",
    "ChargePeriodStart",  # floored to day below
    "ChargeCategory", "ChargeFrequency",
    "ServiceName", "ServiceCategory",
    "RegionId", "AvailabilityZone",
    "ResourceId", "ResourceName", "ResourceType",
    "ConsumedUnit", "PricingUnit", "PricingCategory",
    "SkuId",
    "CommitmentDiscountId", "CommitmentDiscountName",
    "CommitmentDiscountType", "CommitmentDiscountCategory",
    "x_TenantId", "x_TenantName", "x_SourceGranularity",
]


def enrich_tenant(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Join tenant_id / tenant_name onto every row via (Provider, BillingAccountId)."""
    merged = df.merge(
        mapping[["provider", "billing_account_id", "tenant_id", "tenant_name"]],
        left_on=["Provider", "BillingAccountId"],
        right_on=["provider", "billing_account_id"],
        how="left",
    )
    merged["x_TenantId"] = merged["tenant_id"]
    merged["x_TenantName"] = merged["tenant_name"]
    return merged.drop(columns=["provider", "billing_account_id", "tenant_id", "tenant_name"])


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse hourly rows to daily grain by flooring ChargePeriodStart to day."""
    df = df.copy()
    df["ChargePeriodStart"] = df["ChargePeriodStart"].dt.floor("D")
    df["ChargePeriodEnd"] = df["ChargePeriodStart"] + pd.Timedelta(days=1)

    available_keys = [c for c in _DAILY_KEY_COLS if c in df.columns]
    available_measures = [c for c in MEASURE_COLS if c in df.columns]

    agg_dict = {m: "sum" for m in available_measures}
    # Scalar non-key columns: carry first value
    other_cols = [
        c for c in df.columns
        if c not in available_keys
        and c not in available_measures
        and c not in ("ChargePeriodEnd", "x_Metadata")
        and c in FOCUS_COLUMNS
    ]
    for c in other_cols:
        agg_dict[c] = "first"
    agg_dict["x_Metadata"] = "first"
    agg_dict["ChargePeriodEnd"] = "first"

    result = df.groupby(available_keys, dropna=False).agg(agg_dict).reset_index()
    return result[FOCUS_COLUMNS]


def union_all(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True)
