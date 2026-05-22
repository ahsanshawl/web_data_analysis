"""
North Cloud Billing API

Run locally:
    uvicorn api.main:app --reload

Example requests:
    # End-user query (own tenant, daily)
    curl -H "X-Auth-Token: user-token-tenant001" \
         "http://localhost:8000/usage?start=2026-04-16&end=2026-05-01&granularity=day"

    # Admin query — specific tenant, filter by provider
    curl -H "X-Auth-Token: admin-token-northcloud" \
         "http://localhost:8000/usage?start=2026-04-16&end=2026-05-01&tenant_id=tenant_002&provider=GCP"

    # Admin aggregate across all tenants, monthly
    curl -H "X-Auth-Token: admin-token-northcloud" \
         "http://localhost:8000/usage?start=2026-04-01&end=2026-06-01&granularity=month"

    # Anomaly detection (admin)
    curl -H "X-Auth-Token: admin-token-northcloud" \
         "http://localhost:8000/anomalies?start=2026-04-16&end=2026-05-01"
"""

from fastapi import FastAPI, Header, HTTPException, Query
from typing import Annotated

from .auth import resolve_token, enforce_tenant_access, AuthError
from .query import query_usage, query_anomalies, Granularity

app = FastAPI(title="North Cloud Billing API", version="0.1.0")


def _get_identity(x_auth_token: str | None) -> dict:
    if not x_auth_token:
        raise HTTPException(status_code=401, detail="X-Auth-Token header required")
    try:
        return resolve_token(x_auth_token)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.get("/usage")
def get_usage(
    start: str = Query(..., description="Start date inclusive (YYYY-MM-DD)"),
    end: str = Query(..., description="End date exclusive (YYYY-MM-DD)"),
    granularity: Granularity = Query("day", description="Aggregation grain: day | month"),
    provider: str | None = Query(None, description="Filter by provider: AWS | GCP | Azure | Anthropic"),
    service: str | None = Query(None, description="Filter by ServiceName"),
    resource_id: str | None = Query(None, description="Filter by ResourceId"),
    # Admin-only params
    tenant_id: str | None = Query(None, description="Admin only: scope to a single tenant"),
    tenant_ids: str | None = Query(None, description="Admin only: comma-separated tenant list"),
    x_auth_token: Annotated[str | None, Header()] = None,
):
    identity = _get_identity(x_auth_token)

    requested = _parse_tenant_ids(identity, tenant_id, tenant_ids)
    allowed = enforce_tenant_access(identity, requested)

    rows = query_usage(
        tenant_ids=allowed,
        start=start,
        end=end,
        granularity=granularity,
        provider=provider,
        service=service,
        resource_id=resource_id,
    )

    return {
        "meta": {
            "start": start,
            "end": end,
            "granularity": granularity,
            "tenant_ids": allowed,
            "row_count": len(rows),
        },
        "data": rows,
    }


@app.get("/anomalies")
def get_anomalies(
    start: str = Query(..., description="Start date inclusive (YYYY-MM-DD)"),
    end: str = Query(..., description="End date exclusive (YYYY-MM-DD)"),
    provider: str | None = Query(None),
    service: str | None = Query(None),
    tenant_id: str | None = Query(None, description="Admin only"),
    tenant_ids: str | None = Query(None, description="Admin only: comma-separated"),
    x_auth_token: Annotated[str | None, Header()] = None,
):
    identity = _get_identity(x_auth_token)

    requested = _parse_tenant_ids(identity, tenant_id, tenant_ids)
    allowed = enforce_tenant_access(identity, requested)

    rows = query_anomalies(
        tenant_ids=allowed,
        start=start,
        end=end,
        provider=provider,
        service=service,
    )

    return {
        "meta": {
            "start": start,
            "end": end,
            "tenant_ids": allowed,
            "anomaly_count": len(rows),
        },
        "data": rows,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


def _parse_tenant_ids(identity: dict, tenant_id: str | None, tenant_ids: str | None) -> list[str]:
    """Parse admin tenant filter params; reject if called by a non-admin."""
    if (tenant_id or tenant_ids) and identity["role"] != "admin":
        raise HTTPException(status_code=403, detail="tenant_id filter is admin-only")
    if tenant_id:
        return [tenant_id]
    if tenant_ids:
        return [t.strip() for t in tenant_ids.split(",") if t.strip()]
    return []
