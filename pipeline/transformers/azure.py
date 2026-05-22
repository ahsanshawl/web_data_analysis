import json
import re
import pandas as pd

from .base import BaseTransformer

_CHARGE_CATEGORY = {
    "Usage": "Usage",
    "Purchase": "Purchase",
    "Tax": "Tax",
    "UnusedReservation": "Usage",
    "UnusedSavingsPlan": "Usage",
}

_CHARGE_FREQUENCY = {
    "OneTime": "One-Time",
    "Recurring": "Recurring",
    "UsageBased": "Usage-Based",
}

_PRICING_CATEGORY = {
    "OnDemand": "Standard",
    "Spot": "Dynamic",
    "Reservation": "Committed",
    "Savings Plans": "Committed",
}

# Strip leading multiplier (e.g. "100 GB/Month" → "GB/Month", "1 Hour" → "Hour")
_UNIT_PREFIX_RE = re.compile(r"^\d[\d,.]* ?")


def _normalize_unit(raw: str) -> str:
    if not raw or pd.isna(raw):
        return "Units"
    return _UNIT_PREFIX_RE.sub("", str(raw)).strip() or "Units"


def _parse_azure_tags(tag_str) -> dict:
    if not tag_str or pd.isna(tag_str):
        return {}
    result = {}
    for pair in str(tag_str).split(";"):
        pair = pair.strip()
        if ":" in pair:
            k, v = pair.split(":", 1)
            result[k.strip().strip('"')] = v.strip().strip('"')
    return result


def _commitment_category(benefit_id: str) -> str | None:
    if not benefit_id:
        return None
    bid = benefit_id.lower()
    if "/microsoft.capacity/" in bid:
        return "Usage"
    if "/microsoft.billingbenefits/" in bid:
        return "Spend"
    return None


def _commitment_type(benefit_id: str) -> str | None:
    if not benefit_id:
        return None
    bid = benefit_id.lower()
    if "/microsoft.capacity/" in bid:
        return "Reservation"
    if "/microsoft.billingbenefits/" in bid:
        return "Savings Plan"
    return None


class AzureTransformer(BaseTransformer):
    def __init__(self, category_map: dict):
        self._category_map = category_map

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()

        out["BillingAccountId"] = df["billing_account_id"].astype(str)
        out["BillingAccountName"] = df.get("billing_account_name")
        out["SubAccountId"] = df["subscription_id"].astype(str)
        out["SubAccountName"] = df.get("subscription_name")

        out["Provider"] = "Azure"
        out["Publisher"] = df.get("publisher_name", "Microsoft")
        out["InvoiceIssuer"] = "Microsoft"

        out["BillingCurrency"] = df["billing_currency"]
        out["BillingPeriodStart"] = pd.to_datetime(df["billing_period_start_date"], utc=True)
        out["BillingPeriodEnd"] = pd.to_datetime(df["billing_period_end_date"], utc=True)

        # Azure daily export: date column is the charge day
        out["ChargePeriodStart"] = pd.to_datetime(df["date"], utc=True)
        out["ChargePeriodEnd"] = out["ChargePeriodStart"] + pd.Timedelta(days=1)

        out["ChargeCategory"] = df["charge_type"].map(_CHARGE_CATEGORY).fillna("Adjustment")
        out["ChargeDescription"] = df.get("product")
        out["ChargeFrequency"] = df["frequency"].map(_CHARGE_FREQUENCY).fillna("Other")

        out["ServiceName"] = df["meter_category"]
        out["ServiceCategory"] = out["ServiceName"].map(self._category_map).fillna("Other")

        out["RegionId"] = df.get("resource_location")
        out["ResourceId"] = df.get("resource_id")
        out["ResourceName"] = df.get("resource_name")

        out["BilledCost"] = df["cost_in_billing_currency"]
        out["EffectiveCost"] = df["cost_in_billing_currency"]

        out["ConsumedQuantity"] = df["quantity"]
        out["ConsumedUnit"] = df["unit_of_measure"].apply(_normalize_unit)
        out["PricingCategory"] = df["pricing_model"].map(_PRICING_CATEGORY).fillna("Other")

        benefit_id = df.get("benefit_id", pd.Series([""] * len(df))).fillna("")
        out["CommitmentDiscountId"] = benefit_id.where(benefit_id != "", other=None)
        out["CommitmentDiscountName"] = df.get("benefit_name")
        out["CommitmentDiscountCategory"] = benefit_id.apply(_commitment_category)
        out["CommitmentDiscountType"] = benefit_id.apply(_commitment_type)

        out["x_SourceGranularity"] = "daily"
        out["x_Metadata"] = df["tags"].apply(self._build_metadata)

        return self._ensure_schema(out)

    def _build_metadata(self, tag_str) -> str:
        tags = _parse_azure_tags(tag_str)
        meta = {}
        for k, v in tags.items():
            # Normalize known tag keys to shared convention
            normalized = k.lower().replace("-", "_")
            if normalized in ("env", "environment"):
                meta["tag_environment"] = v
            elif normalized == "team":
                meta["tag_team"] = v
            elif normalized in ("costcenter", "cost_center"):
                meta["tag_cost_center"] = v
            else:
                meta[f"tag_{normalized}"] = v
        return json.dumps(meta)
