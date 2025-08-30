from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class AuditLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    COMPLIANCE = "compliance"
    CRITICAL = "critical"


class ActionType(str, Enum):
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    CODE_EXECUTION = "code_execution"
    LLM_REQUEST = "llm_request"
    AGENT_DECISION = "agent_decision"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGE = "config_change"
    SECURITY_EVENT = "security_event"


@dataclass
class AuditEntry:
    audit_id: str
    timestamp: float
    action_type: str
    level: str
    user_id: str
    tenant_id: str
    details: Dict[str, Any]
    context: Dict[str, Any]
    previous_hash: str
    entry_hash: str
    compliance_tags: List[str]


@dataclass
class ComplianceRule:
    rule_id: str
    name: str
    description: str
    pattern: str  # simple substring match for v1
    severity: AuditLevel
    action_required: str  # "log" | "alert" | "block"
    retention_years: int = 7


class ComplianceViolationError(Exception):
    pass


class ImmutableAuditLogger:
    """Lightweight immutable audit logger using storage categories.
    - Immutability is enforced by API (no update/delete endpoints)
    - Chain integrity keeps a running previous_hash/entry_hash sequence stored in audit_meta
    """

    def __init__(self, storage):
        self.storage = storage
        # Load last chain hash if exists
        meta = self.storage.retrieve_memory(key="audit_chain_last", category="audit_meta")
        self._chain_hash = meta[0]["value"] if meta else "genesis"

    def _calc_entry_hash(self, data: Dict[str, Any]) -> str:
        s = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(f"{self._chain_hash}:{s}".encode()).hexdigest()

    def add_rule(self, rule: ComplianceRule) -> bool:
        key = f"compliance_rule_{rule.rule_id}"
        return self.storage.store_memory(key, json.dumps(asdict(rule)), category="compliance_rules")

    def load_rules(self) -> List[ComplianceRule]:
        rules = []
        for row in self.storage.retrieve_memory(category="compliance_rules", limit=500):
            try:
                d = json.loads(row.get("value", "{}"))
                rules.append(ComplianceRule(
                    rule_id=d["rule_id"], name=d["name"], description=d.get("description", ""),
                    pattern=d.get("pattern", ""), severity=AuditLevel(d.get("severity", "info")),
                    action_required=d.get("action_required", "log"), retention_years=int(d.get("retention_years", 7))
                ))
            except Exception:
                continue
        return rules

    def log(self, action_type: ActionType, level: AuditLevel, user_id: Optional[str], tenant_id: Optional[str],
            details: Dict[str, Any], context: Dict[str, Any], compliance_tags: Optional[List[str]] = None) -> str:
        ts = time.time()
        base = {
            "timestamp": ts, "action_type": action_type.value, "level": level.value,
            "user_id": user_id or "system", "tenant_id": tenant_id or "default",
            "details": details or {}, "context": context or {}
        }
        entry_hash = self._calc_entry_hash(base)
        audit_id = f"audit_{int(ts*1000)}"
        entry = AuditEntry(
            audit_id=audit_id, timestamp=ts, action_type=action_type.value, level=level.value,
            user_id=base["user_id"], tenant_id=base["tenant_id"], details=base["details"], context=base["context"],
            previous_hash=self._chain_hash, entry_hash=entry_hash, compliance_tags=compliance_tags or []
        )
        # Persist entry (immutable by convention)
        self.storage.store_memory(audit_id, json.dumps(asdict(entry)), category="audit_entries")
        # Advance chain
        self._chain_hash = entry_hash
        self.storage.store_memory("audit_chain_last", self._chain_hash, category="audit_meta")
        # Simple compliance check (substring)
        for rule in self.load_rules():
            blob = json.dumps(asdict(entry))
            if rule.pattern and rule.pattern.lower() in blob.lower():
                # store alert
                alert_id = f"alert_{int(ts*1000)}_{rule.rule_id}"
                alert = {
                    "alert_id": alert_id, "audit_id": audit_id, "rule_id": rule.rule_id,
                    "severity": rule.severity.value, "message": f"Rule matched: {rule.name}", "created_at": ts
                }
                self.storage.store_memory(alert_id, json.dumps(alert), category="audit_alerts")
                if rule.action_required == "block":
                    raise ComplianceViolationError(f"Blocked by compliance rule: {rule.name}")
        return audit_id

    def verify_chain(self, limit: int = 1000) -> Dict[str, Any]:
        # Fetch recent entries and verify linkage from oldest to newest in the window
        rows = self.storage.retrieve_memory(category="audit_entries", limit=limit)
        entries = []
        for r in rows:
            try:
                entries.append(json.loads(r.get("value", "{}")))
            except Exception:
                continue
        entries.sort(key=lambda e: e.get("timestamp", 0))
        ok = True
        errs: List[str] = []
        prev = "genesis"
        for e in entries:
            calc_src = {k: e.get(k) for k in ['timestamp','action_type','level','user_id','tenant_id','details','context']}
            calc = hashlib.sha256(json.dumps({"prev": prev, "data": calc_src}, sort_keys=True, separators=(',',':')).encode()).hexdigest()
            if e.get("previous_hash") != prev:
                ok = False
                errs.append(f"prev_hash mismatch at {e.get('audit_id')}")
            if e.get("entry_hash") != calc:
                ok = False
                errs.append(f"entry_hash mismatch at {e.get('audit_id')}")
            prev = e.get("entry_hash", prev)
        return {"valid": ok, "errors": errs, "checked": len(entries)}

    def search(self, filters: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        rows = self.storage.retrieve_memory(category="audit_entries", limit=limit)
        out: List[Dict[str, Any]] = []
        for r in rows:
            try:
                e = json.loads(r.get("value", "{}"))
            except Exception:
                continue
            ok = True
            if filters.get("action_type") and e.get("action_type") != filters["action_type"]:
                ok = False
            if filters.get("level") and e.get("level") != filters["level"]:
                ok = False
            if filters.get("tool_name"):
                details = e.get("details", {})
                if (details.get("tool_name") or details.get("tool")) != filters["tool_name"]:
                    ok = False
            if ok:
                out.append(e)
        return out

    def generate_compliance_report(self, tenant_id: Optional[str] = None, start_time: Optional[float] = None, end_time: Optional[float] = None) -> Dict[str, Any]:
        # Simplified: summarize alerts by severity and recent audit entry levels
        alerts = self.storage.retrieve_memory(category="audit_alerts", limit=500)
        by_sev: Dict[str, int] = {}
        for a in alerts:
            try:
                ad = json.loads(a.get("value", "{}"))
                sev = ad.get("severity", "info").lower()
                by_sev[sev] = by_sev.get(sev, 0) + 1
            except Exception:
                continue
        entries = self.storage.retrieve_memory(category="audit_entries", limit=200)
        by_level: Dict[str, int] = {}
        for e in entries:
            try:
                ed = json.loads(e.get("value", "{}"))
                lvl = ed.get("level", "info").lower()
                by_level[lvl] = by_level.get(lvl, 0) + 1
            except Exception:
                continue
        return {"alerts_by_severity": by_sev, "entries_by_level": by_level}


class AttorneyStyleReviewer:
    def __init__(self, audit: ImmutableAuditLogger, server):
        self.audit = audit
        self.server = server

    async def review(self, action_desc: str, context: Dict[str, Any], domain: str = "general") -> Dict[str, Any]:
        prompt = (
            "You are a compliance attorney. Analyze the action for potential legal/regulatory issues. "
            "Return a JSON object with keys: compliance_status (APPROVED|CONDITIONAL|REJECTED), risk_level, "
            "violations (list), recommendations (list), required_actions (list), legal_notes, approval_conditions (list).\n\n"
            f"Action: {action_desc}\nContext: {json.dumps(context, ensure_ascii=False)}\nDomain: {domain}\n"
        )
        try:
            txt = await self.server.make_llm_request_with_retry(prompt, temperature=0.1)
            # Best-effort JSON extraction
            import re
            m = re.search(r"\{[\s\S]*\}", txt)
            data = json.loads(m.group()) if m else {
                "compliance_status": "CONDITIONAL",
                "risk_level": "MEDIUM",
                "violations": [],
                "recommendations": ["Manual review recommended"],
                "required_actions": [],
                "legal_notes": txt,
                "approval_conditions": []
            }
        except Exception as e:
            data = {
                "compliance_status": "REJECTED",
                "risk_level": "CRITICAL",
                "violations": ["Automated review failed"],
                "recommendations": ["Manual legal review required"],
                "required_actions": [],
                "legal_notes": str(e),
                "approval_conditions": []
            }
        # Log the review
        self.audit.log(action_type=ActionType.SECURITY_EVENT, level=AuditLevel.COMPLIANCE,
                       user_id=None, tenant_id=None,
                       details={"action": action_desc, "domain": domain, "status": data.get("compliance_status")},
                       context=context, compliance_tags=["attorney_review", domain])
        return data

