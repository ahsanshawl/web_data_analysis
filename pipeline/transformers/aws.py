import json
import pandas as pd

from .base import BaseTransformer


_CHARGE_CATEGORY = {
    "Tax": "Tax",
    "Fee": "Purchase",
    "SavingsPlanUpfrontFee": "Purchase",
    "RIFee": "Purchase",
    "SavingsPlanRecurringFee": "Purchase",
    "Usage": "Usage",
    "SavingsPlanCoveredUsage": "Usage",
    "SavingsPlanNegation": "Usage",
    "DiscountedUsage": "Usage",
    "BundledDiscount": "Usage",
    "Discount": "Usage",
    "PrivateRateDiscount": "Usage",
    "EdpDiscount": "Usage",
    "Credit": "Adjustment",
    "Refund": "Adjustment",
}

_CHARGE_FREQUENCY = {
    "Refund": "One-Time",
    "Purchase": "One-Time",
    "Anniversary": "Recurring",
}

_PRICING_CATEGORY = {
    "On-Demand": "Standard",
    "Reserved Instances": "Committed",
    "Spot Instances": "Dynamic",
    "Dedicated Hosts": "Standard",
}


def _col(df: pd.DataFrame, name: str, default=None):
    return df[name] if name in df.columns else default


class AWSTransformer(BaseTransformer):
    def __init__(self, category_map: dict):
        self._category_map = category_map

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()

        out["BillingAccountId"] = df["bill_payer_account_id"].astype(str)
        out["SubAccountId"] = _col(df, "line_item_usage_account_id", pd.NA)

        out["Provider"] = "AWS"
        out["Publisher"] = _col(df, "line_item_legal_entity", "AWS")
        out["InvoiceIssuer"] = _col(df, "bill_invoicing_entity", "AWS")

        out["BillingCurrency"] = _col(df, "line_item_currency_code", "USD")
        out["BillingPeriodStart"] = pd.to_datetime(
            _col(df, "bill_billing_period_start_date"), utc=True
        )
        out["BillingPeriodEnd"] = pd.to_datetime(
            _col(df, "bill_billing_period_end_date"),
            utc=True,
            errors="coerce",
        )

        out["ChargePeriodStart"] = pd.to_datetime(
            df["line_item_usage_start_date"], utc=True
        )
        out["ChargePeriodEnd"] = pd.to_datetime(
            df["line_item_usage_end_date"], utc=True
        )

        line_type = _col(df, "line_item_line_item_type", pd.Series(["Usage"] * len(df)))
        out["ChargeCategory"] = line_type.map(_CHARGE_CATEGORY).fillna("")
        out["ChargeDescription"] = _col(df, "line_item_line_item_description")
        out["ChargeFrequency"] = _col(df, "bill_bill_type", pd.Series([""] * len(df))).map(
            _CHARGE_FREQUENCY
        ).fillna("")

        out["ServiceName"] = _col(df, "line_item_product_code")
        out["ServiceCategory"] = out["ServiceName"].map(self._category_map).fillna("Other")

        region = _col(df, "product_region")
        if region is None:
            region = _col(df, "product_region_code")
        out["RegionId"] = region
        out["AvailabilityZone"] = _col(df, "line_item_availability_zone")

        resource_id = _col(df, "line_item_resource_id")
        out["ResourceId"] = resource_id
        out["ResourceName"] = resource_id.str.split(":").str.get(6).str.title() if resource_id is not None else None
        out["ResourceType"] = resource_id.str.split(":").str.get(5) if resource_id is not None else None

        net_cost = _col(df, "line_item_net_unblended_cost")
        unblended = df["line_item_unblended_cost"]
        out["BilledCost"] = net_cost.where(net_cost.notna(), unblended) if net_cost is not None else unblended
        out["EffectiveCost"] = out["BilledCost"]  # simplified: no savings plan data in sample

        list_rate = _col(df, "pricing_public_on_demand_rate")
        usage_amount = _col(df, "line_item_usage_amount", pd.Series([0.0] * len(df)))
        out["ListUnitPrice"] = list_rate
        out["ListCost"] = (
            list_rate * usage_amount if list_rate is not None
            else _col(df, "pricing_public_on_demand_cost")
        )

        out["ConsumedQuantity"] = _col(df, "line_item_usage_amount")
        out["ConsumedUnit"] = _col(df, "pricing_unit")
        out["PricingQuantity"] = _col(df, "line_item_usage_amount")
        out["PricingUnit"] = _col(df, "pricing_unit")

        purchase_option = _col(df, "product_purchase_option", pd.Series([""] * len(df)))
        out["PricingCategory"] = purchase_option.map(_PRICING_CATEGORY).fillna("Other")

        out["SkuId"] = _col(df, "product_sku")
        sku_price = _col(df, "pricing_rate_code")
        if sku_price is None:
            sku_price = _col(df, "pricing_rate_id")
        out["SkuPriceId"] = sku_price

        savings_arn = _col(df, "savings_plan_savings_plan_arn")
        reservation_arn = _col(df, "reservation_reservation_arn")
        if savings_arn is not None:
            out["CommitmentDiscountId"] = savings_arn.where(savings_arn.notna(), reservation_arn)
            out["CommitmentDiscountType"] = savings_arn.where(savings_arn.notna(), "").apply(
                lambda v: "Savings Plan" if v else ("Reserved Instances (RI)" if reservation_arn is not None else None)
            )
            out["CommitmentDiscountCategory"] = savings_arn.where(savings_arn.notna(), "").apply(
                lambda v: "Spend" if v else ("Usage" if reservation_arn is not None else None)
            )

        out["x_SourceGranularity"] = "hourly"
        out["x_Metadata"] = df.apply(self._build_metadata, axis=1)

        return self._ensure_schema(out)

    def _build_metadata(self, row) -> str:
        meta = {}
        for tag_col, key in [
            ("resource_tags_user_environment", "tag_environment"),
            ("resource_tags_user_team", "tag_team"),
            ("resource_tags_user_cost_center", "tag_cost_center"),
        ]:
            val = row.get(tag_col)
            if val and str(val) not in ("nan", "None", ""):
                meta[key] = str(val)
        return json.dumps(meta)
