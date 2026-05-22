from abc import ABC, abstractmethod
import pandas as pd

FOCUS_COLUMNS = [
    # Account hierarchy
    "BillingAccountId", "BillingAccountName",
    "SubAccountId", "SubAccountName",
    # Provider chain
    "Provider", "Publisher", "InvoiceIssuer",
    # Currency & billing period
    "BillingCurrency", "BillingPeriodStart", "BillingPeriodEnd",
    # Charge period (daily after normalization)
    "ChargePeriodStart", "ChargePeriodEnd",
    # Charge classification
    "ChargeCategory", "ChargeDescription", "ChargeFrequency",
    # Service
    "ServiceName", "ServiceCategory",
    # Geography
    "RegionId", "AvailabilityZone",
    # Resource
    "ResourceId", "ResourceName", "ResourceType",
    # Cost measures
    "BilledCost", "EffectiveCost", "ListCost", "ListUnitPrice",
    # Usage
    "ConsumedQuantity", "ConsumedUnit",
    # Pricing
    "PricingQuantity", "PricingUnit", "PricingCategory",
    # SKU
    "SkuId", "SkuPriceId",
    # Commitment discounts
    "CommitmentDiscountId", "CommitmentDiscountName",
    "CommitmentDiscountType", "CommitmentDiscountCategory",
    # Custom extensions
    "x_TenantId", "x_TenantName",
    "x_SourceGranularity",
    "x_Metadata",  # JSON: tags + provider-specific attribution fields
]

MEASURE_COLS = {"BilledCost", "EffectiveCost", "ListCost", "ConsumedQuantity", "PricingQuantity"}


class BaseTransformer(ABC):
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map source DataFrame to the canonical FOCUS schema."""

    def _ensure_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add any missing FOCUS columns as None and enforce column order."""
        for col in FOCUS_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[FOCUS_COLUMNS]
