import json
import pandas as pd
import pytz

from .base import BaseTransformer

_LA_TZ = pytz.timezone("America/Los_Angeles")

_CHARGE_CATEGORY = {
    "regular": "Usage",
    "tax": "Tax",
    "adjustment": "Adjustment",
    "credit": "Adjustment",
    "rounding_error": "Adjustment",
}


def _unnest(df: pd.DataFrame, struct_col: str, field: str, out_col: str):
    df[out_col] = df[struct_col].apply(
        lambda v: v.get(field) if isinstance(v, dict) else None
    )


def _sum_credits(credits) -> float:
    if credits is None:
        return 0.0
    try:
        return sum(c.get("amount", 0) for c in credits if isinstance(c, dict))
    except TypeError:
        return 0.0


def _extract_labels(labels) -> dict:
    result = {}
    if labels is None:
        return result
    try:
        for item in labels:
            if isinstance(item, dict):
                result[item.get("key", "")] = item.get("value", "")
    except TypeError:
        pass
    return result


def _parse_billing_month(month_str: str) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(str(month_str), tz=_LA_TZ)
        return ts.tz_convert("UTC")
    except Exception:
        return pd.NaT


class GCPTransformer(BaseTransformer):
    def __init__(self, category_map: dict):
        self._category_map = category_map

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()

        out["BillingAccountId"] = df["billing_account_id"].astype(str)

        # Flatten project struct for SubAccountId / SubAccountName
        out["SubAccountId"] = df["project"].apply(
            lambda v: v.get("id") if isinstance(v, dict) else None
        )
        out["SubAccountName"] = df["project"].apply(
            lambda v: v.get("name") if isinstance(v, dict) else None
        )

        out["Provider"] = "GCP"
        out["Publisher"] = "Google Cloud"
        out["InvoiceIssuer"] = "Google Cloud"

        out["BillingCurrency"] = df["currency"]

        # invoice.month → YYYYMM → first of month (LA tz → UTC)
        invoice_month = df["invoice"].apply(
            lambda v: v.get("month") if isinstance(v, dict) else None
        )
        out["BillingPeriodStart"] = invoice_month.apply(_parse_billing_month)
        out["BillingPeriodEnd"] = out["BillingPeriodStart"].apply(
            lambda t: t + pd.offsets.MonthEnd(1) if pd.notna(t) else pd.NaT
        )

        out["ChargePeriodStart"] = pd.to_datetime(df["usage_start_time"], utc=True)
        out["ChargePeriodEnd"] = pd.to_datetime(df["usage_end_time"], utc=True)

        out["ChargeCategory"] = df["cost_type"].map(_CHARGE_CATEGORY).fillna("Usage")

        # Service
        _unnest(df, "service", "description", "_service_desc")
        out["ServiceName"] = df["_service_desc"]
        out["ServiceCategory"] = out["ServiceName"].map(self._category_map).fillna("Other")
        df.drop(columns=["_service_desc"], inplace=True, errors="ignore")

        # Location
        out["RegionId"] = df["location"].apply(
            lambda v: v.get("region") or v.get("location") if isinstance(v, dict) else None
        )
        out["AvailabilityZone"] = df["location"].apply(
            lambda v: v.get("zone") if isinstance(v, dict) else None
        )

        # Resource
        out["ResourceId"] = df["resource"].apply(
            lambda v: v.get("global_name") if isinstance(v, dict) else None
        )
        out["ResourceName"] = df["resource"].apply(
            lambda v: v.get("name") if isinstance(v, dict) else None
        )

        # Cost
        out["BilledCost"] = df["cost"]
        credits_sum = df["credits"].apply(_sum_credits)
        effective = df["cost"] + credits_sum
        out["EffectiveCost"] = effective.clip(lower=0)

        # SKU
        _unnest(df, "sku", "id", "_sku_id")
        out["SkuId"] = df["_sku_id"]
        df.drop(columns=["_sku_id"], inplace=True, errors="ignore")

        # Usage
        out["ConsumedQuantity"] = df["usage"].apply(
            lambda v: v.get("amount") if isinstance(v, dict) else None
        )
        out["ConsumedUnit"] = df["usage"].apply(
            lambda v: v.get("unit") if isinstance(v, dict) else None
        )
        out["PricingQuantity"] = df["usage"].apply(
            lambda v: v.get("amount_in_pricing_units") if isinstance(v, dict) else None
        )
        out["PricingUnit"] = df["usage"].apply(
            lambda v: v.get("pricing_unit") if isinstance(v, dict) else None
        )

        out["x_SourceGranularity"] = "hourly"
        out["x_Metadata"] = df.apply(self._build_metadata, axis=1)

        return self._ensure_schema(out)

    def _build_metadata(self, row) -> str:
        meta = {}
        labels = _extract_labels(row.get("labels"))
        for k, v in labels.items():
            tag_key = f"tag_{k}" if not k.startswith("tag_") else k
            meta[tag_key] = v
        return json.dumps(meta)
