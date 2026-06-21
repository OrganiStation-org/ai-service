import json
from fastapi import Header, HTTPException


def require_ai_admin(x_user_permissions: str = Header(default="[]")) -> None:
    """Reject document management unless the caller has ai:admin."""
    try:
        permissions = json.loads(x_user_permissions or "[]")
    except json.JSONDecodeError:
        permissions = []

    if "ai:admin" not in permissions:
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to manage AI documents.",
        )
