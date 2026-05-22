"""
Token → {tenant_id, role} lookup.

In production this would be a secrets store or JWT. Here it's hardcoded.
role: "user"  — can only read their own tenant
role: "admin" — can read any tenant or aggregate across tenants
"""

_TOKENS: dict[str, dict] = {
    "user-token-tenant001": {"tenant_id": "tenant_001", "role": "user"},
    "user-token-tenant002": {"tenant_id": "tenant_002", "role": "user"},
    "admin-token-northcloud": {"tenant_id": None, "role": "admin"},
}


class AuthError(Exception):
    pass


class ForbiddenError(Exception):
    pass


def resolve_token(token: str) -> dict:
    identity = _TOKENS.get(token)
    if not identity:
        raise AuthError("Invalid or missing X-Auth-Token")
    return identity


def enforce_tenant_access(identity: dict, requested_tenant_ids: list[str]) -> list[str]:
    """
    Return the tenant IDs the caller is actually allowed to query.
    - Users: always scoped to their own tenant_id; requested list is ignored.
    - Admins: may pass an explicit list, or get all known tenants if none given.
    """
    if identity["role"] == "user":
        return [identity["tenant_id"]]

    # Admin path
    if requested_tenant_ids:
        return requested_tenant_ids

    # Admin with no filter → all known tenants
    return list({v["tenant_id"] for v in _TOKENS.values() if v["tenant_id"]})
