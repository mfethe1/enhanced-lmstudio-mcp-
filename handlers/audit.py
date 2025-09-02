from __future__ import annotations

import asyncio
from typing import Any, Dict

from server import (
    AuditLevel,
    ComplianceRule,
)


def _require_audit():
    from server import _require_audit as _ra
    return _ra()


def handle_audit_search(arguments: Dict[str, Any], server) -> str:
    audit = _require_audit()
    filters = {k: arguments.get(k) for k in ("action_type", "level", "tool_name") if arguments.get(k)}
    limit = int(arguments.get("limit", 100))
    return audit.search(filters, limit=limit)


def handle_audit_verify_integrity(arguments: Dict[str, Any], server) -> str:
    audit = _require_audit()
    limit = int(arguments.get("limit", 200))
    return audit.verify_chain(limit=limit)


def handle_audit_add_rule(arguments: Dict[str, Any], server) -> Dict[str, Any]:
    audit = _require_audit()
    rid = (arguments.get("rule_id") or "").strip()
    name = (arguments.get("name") or "").strip()
    pattern = (arguments.get("pattern") or "").strip()
    if not rid or not name or not pattern:
        raise Exception("rule_id, name, and pattern are required")
    severity_str = (arguments.get("severity") or "info").lower()
    try:
        sev = AuditLevel(severity_str)
    except Exception:
        sev = AuditLevel.INFO
    action_required = (arguments.get("action_required") or "log").lower()
    retention_years = int(arguments.get("retention_years", 7))
    rule = ComplianceRule(
        rule_id=rid,
        name=name,
        description=arguments.get("description", ""),
        pattern=pattern,
        severity=sev,
        action_required=action_required,
        retention_years=retention_years,
    )
    audit.add_rule(rule)
    return {"ok": True}


def handle_audit_compliance_report(arguments: Dict[str, Any], server) -> Dict[str, Any]:
    audit = _require_audit()
    return audit.generate_compliance_report(tenant_id=arguments.get("tenant_id"))


def handle_audit_review_action(arguments: Dict[str, Any], server) -> str:
    audit = _require_audit()
    action_desc = (arguments.get("action_desc") or "").strip()
    if not action_desc:
        raise Exception("action_desc is required")
    context = arguments.get("context") or {}
    domain = (arguments.get("domain") or "general").strip()
    from server import AttorneyStyleReviewer
    reviewer = AttorneyStyleReviewer(audit, server)
    try:
        return asyncio.get_event_loop().run_until_complete(reviewer.review(action_desc, context, domain))
    except RuntimeError:
        loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(reviewer.review(action_desc, context, domain))
        finally:
            loop.close()

