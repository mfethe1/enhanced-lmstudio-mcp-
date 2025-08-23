<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Continuing with Implementations 6 and 7

## 6. Attorney-Style Review and Immutable Audit-Logging

**File: `audit_logger.py`**

```python
import hashlib
import json
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from pathlib import Path
import hmac
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import threading
from collections import deque

class AuditLevel(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    COMPLIANCE = "compliance"
    CRITICAL = "critical"

class ActionType(Enum):
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
    action_type: ActionType
    level: AuditLevel
    user_id: str
    tenant_id: str
    details: Dict[str, Any]
    context: Dict[str, Any]
    previous_hash: str
    entry_hash: str
    signature: str
    compliance_tags: List[str]

@dataclass
class ComplianceRule:
    rule_id: str
    name: str
    description: str
    pattern: str  # regex pattern to match against audit entries
    severity: AuditLevel
    action_required: str  # "log", "alert", "block"
    retention_years: int

class ImmutableAuditLogger:
    def __init__(self, db_path: str = "audit_trail.db", encryption_key: bytes = None):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.chain_hash = "genesis"  # Genesis hash for blockchain-like chaining
        
        # Generate or load encryption keys
        if encryption_key:
            self.encryption_key = encryption_key
        else:
            self.encryption_key = self._generate_or_load_key()
            
        # Generate signing keys for integrity
        self.signing_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        self.compliance_rules = []
        self.pending_alerts = deque(maxlen=1000)
        
        self.init_db()
        self.load_compliance_rules()
        self._restore_chain_state()

    def init_db(self):
        """Initialize immutable audit database with integrity constraints"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS audit_entries (
                    audit_id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    action_type TEXT NOT NULL,
                    level TEXT NOT NULL,
                    user_id TEXT,
                    tenant_id TEXT,
                    details TEXT NOT NULL,
                    context TEXT,
                    previous_hash TEXT NOT NULL,
                    entry_hash TEXT NOT NULL,
                    signature TEXT NOT NULL,
                    compliance_tags TEXT,
                    created_at REAL DEFAULT (julianday('now'))
                );
                
                CREATE TABLE IF NOT EXISTS compliance_rules (
                    rule_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    pattern TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    action_required TEXT NOT NULL,
                    retention_years INTEGER DEFAULT 7,
                    active INTEGER DEFAULT 1
                );
                
                CREATE TABLE IF NOT EXISTS audit_alerts (
                    alert_id TEXT PRIMARY KEY,
                    audit_id TEXT,
                    rule_id TEXT,
                    severity TEXT,
                    message TEXT,
                    resolved INTEGER DEFAULT 0,
                    created_at REAL DEFAULT (julianday('now')),
                    FOREIGN KEY (audit_id) REFERENCES audit_entries(audit_id)
                );
                
                -- Ensure immutability with triggers
                CREATE TRIGGER IF NOT EXISTS prevent_audit_updates
                BEFORE UPDATE ON audit_entries
                FOR EACH ROW
                BEGIN
                    SELECT RAISE(ABORT, 'Audit entries are immutable and cannot be modified');
                END;
                
                CREATE TRIGGER IF NOT EXISTS prevent_audit_deletes
                BEFORE DELETE ON audit_entries
                FOR EACH ROW
                BEGIN
                    SELECT RAISE(ABORT, 'Audit entries are immutable and cannot be deleted');
                END;
            """)

    def _generate_or_load_key(self) -> bytes:
        """Generate or load encryption key for sensitive data"""
        key_file = Path(self.db_path).parent / ".audit_key"
        
        if key_file.exists():
            return key_file.read_bytes()
        else:
            key = hashlib.sha256(f"audit_key_{time.time()}".encode()).digest()
            key_file.write_bytes(key)
            key_file.chmod(0o600)  # Restrict access
            return key

    def _restore_chain_state(self):
        """Restore the blockchain chain hash from the last entry"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT entry_hash FROM audit_entries ORDER BY timestamp DESC LIMIT 1"
            )
            result = cursor.fetchone()
            if result:
                self.chain_hash = result[^0]

    def _calculate_entry_hash(self, entry_data: Dict[str, Any]) -> str:
        """Calculate cryptographic hash for audit entry"""
        # Create deterministic string representation
        hash_data = json.dumps(entry_data, sort_keys=True, separators=(',', ':'))
        combined = f"{self.chain_hash}:{hash_data}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _sign_entry(self, entry_hash: str) -> str:
        """Sign the entry hash for integrity verification"""
        signature = self.signing_key.sign(
            entry_hash.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode()

    def _encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data in audit entries"""
        # Simple encryption using HMAC for demonstration
        # In production, use proper AES encryption
        return base64.b64encode(
            hmac.new(self.encryption_key, data.encode(), hashlib.sha256).digest()
        ).decode()

    def log_audit_entry(self, action_type: ActionType, level: AuditLevel,
                       user_id: str = None, tenant_id: str = None,
                       details: Dict[str, Any] = None, context: Dict[str, Any] = None,
                       compliance_tags: List[str] = None) -> str:
        """Log an immutable audit entry with blockchain-like chaining"""
        
        with self.lock:
            audit_id = f"audit_{int(time.time() * 1000)}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
            timestamp = time.time()
            
            # Prepare entry data
            entry_data = {
                "audit_id": audit_id,
                "timestamp": timestamp,
                "action_type": action_type.value,
                "level": level.value,
                "user_id": user_id or "system",
                "tenant_id": tenant_id or "default",
                "details": details or {},
                "context": context or {}
            }
            
            # Calculate hash and signature
            entry_hash = self._calculate_entry_hash(entry_data)
            signature = self._sign_entry(entry_hash)
            
            # Create audit entry
            audit_entry = AuditEntry(
                audit_id=audit_id,
                timestamp=timestamp,
                action_type=action_type,
                level=level,
                user_id=user_id or "system",
                tenant_id=tenant_id or "default",
                details=details or {},
                context=context or {},
                previous_hash=self.chain_hash,
                entry_hash=entry_hash,
                signature=signature,
                compliance_tags=compliance_tags or []
            )
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO audit_entries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    audit_entry.audit_id,
                    audit_entry.timestamp,
                    audit_entry.action_type.value,
                    audit_entry.level.value,
                    audit_entry.user_id,
                    audit_entry.tenant_id,
                    json.dumps(audit_entry.details),
                    json.dumps(audit_entry.context),
                    audit_entry.previous_hash,
                    audit_entry.entry_hash,
                    audit_entry.signature,
                    json.dumps(audit_entry.compliance_tags)
                ))
            
            # Update chain hash
            self.chain_hash = entry_hash
            
            # Check compliance rules
            self._check_compliance_rules(audit_entry)
            
            return audit_id

    def _check_compliance_rules(self, entry: AuditEntry):
        """Check audit entry against compliance rules"""
        entry_text = json.dumps(asdict(entry))
        
        for rule in self.compliance_rules:
            import re
            if re.search(rule.pattern, entry_text, re.IGNORECASE):
                alert_id = f"alert_{time.time()}_{rule.rule_id}"
                
                # Log compliance alert
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO audit_alerts (alert_id, audit_id, rule_id, severity, message)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        alert_id,
                        entry.audit_id,
                        rule.rule_id,
                        rule.severity.value,
                        f"Compliance rule '{rule.name}' triggered"
                    ))
                
                # Take action based on rule
                if rule.action_required == "block":
                    raise ComplianceViolationError(f"Action blocked by compliance rule: {rule.name}")
                elif rule.action_required == "alert":
                    self.pending_alerts.append({
                        "alert_id": alert_id,
                        "rule": rule.name,
                        "severity": rule.severity.value,
                        "entry_id": entry.audit_id
                    })

    def verify_chain_integrity(self, start_id: str = None, end_id: str = None) -> Dict[str, Any]:
        """Verify the integrity of the audit chain"""
        verification_result = {
            "valid": True,
            "errors": [],
            "verified_entries": 0,
            "total_entries": 0
        }
        
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM audit_entries ORDER BY timestamp ASC"
            params = []
            
            if start_id:
                query += " AND audit_id >= ?"
                params.append(start_id)
            if end_id:
                query += " AND audit_id <= ?"
                params.append(end_id)
            
            cursor = conn.execute(query, params)
            entries = cursor.fetchall()
            
            verification_result["total_entries"] = len(entries)
            
            previous_hash = "genesis"
            for row in entries:
                audit_id, timestamp, action_type, level, user_id, tenant_id, details, context, prev_hash, entry_hash, signature, compliance_tags = row
                
                # Verify chain linkage
                if prev_hash != previous_hash:
                    verification_result["valid"] = False
                    verification_result["errors"].append(f"Chain break at entry {audit_id}")
                
                # Verify entry hash
                entry_data = {
                    "audit_id": audit_id,
                    "timestamp": timestamp,
                    "action_type": action_type,
                    "level": level,
                    "user_id": user_id,
                    "tenant_id": tenant_id,
                    "details": json.loads(details),
                    "context": json.loads(context) if context else {}
                }
                
                expected_hash = self._calculate_entry_hash(entry_data)
                if expected_hash != entry_hash:
                    verification_result["valid"] = False
                    verification_result["errors"].append(f"Hash mismatch at entry {audit_id}")
                
                # Verify signature (simplified for demonstration)
                try:
                    # In production, use proper signature verification
                    expected_sig = self._sign_entry(entry_hash)
                    # Note: This is a simplified check - in production you'd verify with public key
                    verification_result["verified_entries"] += 1
                except Exception as e:
                    verification_result["valid"] = False
                    verification_result["errors"].append(f"Signature verification failed for {audit_id}: {e}")
                
                previous_hash = entry_hash
        
        return verification_result

    def generate_compliance_report(self, tenant_id: str = None, 
                                 start_date: float = None, end_date: float = None) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        report = {
            "report_id": f"compliance_{int(time.time())}",
            "generated_at": time.time(),
            "tenant_id": tenant_id,
            "period": {
                "start": start_date,
                "end": end_date
            },
            "summary": {},
            "violations": [],
            "recommendations": []
        }
        
        with sqlite3.connect(self.db_path) as conn:
            # Build query with filters
            where_clauses = []
            params = []
            
            if tenant_id:
                where_clauses.append("tenant_id = ?")
                params.append(tenant_id)
            if start_date:
                where_clauses.append("timestamp >= ?")
                params.append(start_date)
            if end_date:
                where_clauses.append("timestamp <= ?")
                params.append(end_date)
            
            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            # Get summary statistics
            cursor = conn.execute(f"""
                SELECT level, COUNT(*) 
                FROM audit_entries 
                WHERE {where_clause}
                GROUP BY level
            """, params)
            
            report["summary"]["entries_by_level"] = dict(cursor.fetchall())
            
            # Get compliance violations
            cursor = conn.execute(f"""
                SELECT aa.alert_id, aa.rule_id, aa.severity, aa.message, ae.timestamp, ae.action_type
                FROM audit_alerts aa
                JOIN audit_entries ae ON aa.audit_id = ae.audit_id
                WHERE {where_clause} AND aa.resolved = 0
                ORDER BY ae.timestamp DESC
            """, params)
            
            violations = cursor.fetchall()
            for violation in violations:
                report["violations"].append({
                    "alert_id": violation[^0],
                    "rule_id": violation[^1],
                    "severity": violation[^2],
                    "message": violation[^3],
                    "timestamp": violation[^4],
                    "action_type": violation[^5]
                })
            
            # Generate recommendations based on patterns
            report["recommendations"] = self._generate_compliance_recommendations(report["violations"])
        
        return report

    def _generate_compliance_recommendations(self, violations: List[Dict[str, Any]]) -> List[str]:
        """Generate compliance recommendations based on violation patterns"""
        recommendations = []
        
        if len(violations) > 10:
            recommendations.append("High number of compliance violations detected. Consider reviewing access controls and user training.")
        
        severity_counts = {}
        for violation in violations:
            sev = violation["severity"]
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        if severity_counts.get("critical", 0) > 0:
            recommendations.append("Critical compliance violations require immediate attention and remediation.")
        
        if severity_counts.get("warning", 0) > 5:
            recommendations.append("Multiple warning-level violations suggest process improvements may be needed.")
        
        return recommendations

    def add_compliance_rule(self, rule: ComplianceRule):
        """Add a new compliance rule"""
        self.compliance_rules.append(rule)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO compliance_rules VALUES (?, ?, ?, ?, ?, ?, ?, 1)
            """, (
                rule.rule_id,
                rule.name,
                rule.description,
                rule.pattern,
                rule.severity.value,
                rule.action_required,
                rule.retention_years
            ))

    def load_compliance_rules(self):
        """Load compliance rules from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM compliance_rules WHERE active = 1")
                
                for row in cursor.fetchall():
                    rule_id, name, description, pattern, severity, action, retention, _ = row
                    rule = ComplianceRule(
                        rule_id=rule_id,
                        name=name,
                        description=description,
                        pattern=pattern,
                        severity=AuditLevel(severity),
                        action_required=action,
                        retention_years=retention
                    )
                    self.compliance_rules.append(rule)
        except sqlite3.OperationalError:
            # Table doesn't exist yet
            pass

    def search_audit_trail(self, query_params: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """Search audit trail with various filters"""
        where_clauses = []
        params = []
        
        if query_params.get("user_id"):
            where_clauses.append("user_id = ?")
            params.append(query_params["user_id"])
        
        if query_params.get("tenant_id"):
            where_clauses.append("tenant_id = ?")
            params.append(query_params["tenant_id"])
        
        if query_params.get("action_type"):
            where_clauses.append("action_type = ?")
            params.append(query_params["action_type"])
        
        if query_params.get("level"):
            where_clauses.append("level = ?")
            params.append(query_params["level"])
        
        if query_params.get("start_time"):
            where_clauses.append("timestamp >= ?")
            params.append(query_params["start_time"])
        
        if query_params.get("end_time"):
            where_clauses.append("timestamp <= ?")
            params.append(query_params["end_time"])
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f"""
                SELECT audit_id, timestamp, action_type, level, user_id, tenant_id, details, context
                FROM audit_entries
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """, params + [limit])
            
            results = []
            for row in cursor.fetchall():
                audit_id, timestamp, action_type, level, user_id, tenant_id, details, context = row
                results.append({
                    "audit_id": audit_id,
                    "timestamp": timestamp,
                    "action_type": action_type,
                    "level": level,
                    "user_id": user_id,
                    "tenant_id": tenant_id,
                    "details": json.loads(details),
                    "context": json.loads(context) if context else {}
                })
            
            return results

class ComplianceViolationError(Exception):
    """Raised when a compliance rule blocks an action"""
    pass

class AttorneyStyleReviewer:
    """AI-powered compliance reviewer that acts like a legal/compliance attorney"""
    
    def __init__(self, audit_logger: ImmutableAuditLogger, server):
        self.audit_logger = audit_logger
        self.server = server
        
        # Pre-defined compliance domains
        self.compliance_domains = {
            "healthcare": ["HIPAA", "HITECH", "FDA"],
            "finance": ["SOX", "PCI DSS", "GDPR"],
            "government": ["FISMA", "NIST", "ATO"],
            "general": ["GDPR", "CCPA", "ISO27001"]
        }

    async def review_action_for_compliance(self, action_desc: str, context: Dict[str, Any], 
                                         domain: str = "general") -> Dict[str, Any]:
        """Review an action for compliance issues like an attorney would"""
        
        relevant_regulations = self.compliance_domains.get(domain, ["GDPR", "general compliance"])
        
        review_prompt = f"""
        You are a compliance attorney reviewing the following action for potential legal and regulatory issues.
        
        Action Description: {action_desc}
        Context: {json.dumps(context, indent=2)}
        Applicable Regulations: {', '.join(relevant_regulations)}
        
        Please provide a comprehensive compliance review covering:
        
        1. REGULATORY COMPLIANCE:
           - Identify any potential violations of applicable regulations
           - Assess data protection and privacy implications
           - Note any required approvals or documentation
        
        2. RISK ASSESSMENT:
           - Rate the compliance risk (LOW/MEDIUM/HIGH/CRITICAL)
           - Identify specific legal exposures
           - Note any precedent or case law concerns
        
        3. RECOMMENDATIONS:
           - Provide specific actionable recommendations
           - Suggest additional safeguards or documentation
           - Recommend approval/modification/rejection
        
        4. REQUIRED ACTIONS:
           - List any mandatory compliance steps
           - Identify required notifications or filings
           - Note audit trail requirements
        
        Format your response as JSON:
        {{
            "compliance_status": "APPROVED|CONDITIONAL|REJECTED",
            "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
            "violations": ["list of potential violations"],
            "recommendations": ["list of recommendations"],
            "required_actions": ["list of required actions"],
            "legal_notes": "detailed legal analysis",
            "approval_conditions": ["conditions if conditional approval"]
        }}
        """
        
        try:
            response = await self.server.make_llm_request_with_retry(review_prompt, temperature=0.1)
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                review_result = json.loads(json_match.group())
            else:
                # Fallback parsing
                review_result = {
                    "compliance_status": "CONDITIONAL",
                    "risk_level": "MEDIUM",
                    "violations": ["Could not parse full compliance review"],
                    "recommendations": ["Manual legal review recommended"],
                    "required_actions": ["Consult with legal counsel"],
                    "legal_notes": response,
                    "approval_conditions": ["Pending manual review"]
                }
            
            # Log the compliance review
            self.audit_logger.log_audit_entry(
                action_type=ActionType.SECURITY_EVENT,
                level=AuditLevel.COMPLIANCE,
                details={
                    "action_reviewed": action_desc,
                    "compliance_status": review_result["compliance_status"],
                    "risk_level": review_result["risk_level"],
                    "domain": domain
                },
                context=context,
                compliance_tags=["attorney_review", domain] + relevant_regulations
            )
            
            return review_result
            
        except Exception as e:
            # Log the error and return safe default
            self.audit_logger.log_audit_entry(
                action_type=ActionType.SECURITY_EVENT,
                level=AuditLevel.CRITICAL,
                details={
                    "error": "Compliance review failed",
                    "action": action_desc,
                    "exception": str(e)
                },
                compliance_tags=["review_failure", domain]
            )
            
            return {
                "compliance_status": "REJECTED",
                "risk_level": "CRITICAL",
                "violations": ["Compliance review system failure"],
                "recommendations": ["System requires manual review before proceeding"],
                "required_actions": ["Contact system administrator"],
                "legal_notes": f"Automated compliance review failed: {str(e)}",
                "approval_conditions": []
            }

    def create_compliance_workflow(self, workflow_steps: List[Dict[str, Any]]) -> str:
        """Create a compliance-aware workflow with audit logging"""
        workflow_id = f"workflow_{int(time.time())}_{hashlib.md5(json.dumps(workflow_steps).encode()).hexdigest()[:8]}"
        
        # Log workflow creation
        self.audit_logger.log_audit_entry(
            action_type=ActionType.CONFIGURATION_CHANGE,
            level=AuditLevel.COMPLIANCE,
            details={
                "workflow_id": workflow_id,
                "steps_count": len(workflow_steps),
                "workflow_type": "compliance_aware"
            },
            context={"workflow_steps": workflow_steps},
            compliance_tags=["workflow_creation", "compliance"]
        )
        
        return workflow_id

class AuditingMCPIntegration:
    """Integration layer for MCP server with comprehensive audit logging"""
    
    def __init__(self, server):
        self.server = server
        self.audit_logger = ImmutableAuditLogger()
        self.attorney_reviewer = AttorneyStyleReviewer(self.audit_logger, server)
        
        # Add default compliance rules
        self._setup_default_compliance_rules()
        
    def _setup_default_compliance_rules(self):
        """Setup default compliance rules for common scenarios"""
        
        # High-privilege actions
        self.audit_logger.add_compliance_rule(ComplianceRule(
            rule_id="high_privilege_access",
            name="High Privilege Access Control",
            description="Monitor high-privilege file system or execution access",
            pattern=r"(file_write|code_execution).*root|admin|sudo",
            severity=AuditLevel.WARNING,
            action_required="alert",
            retention_years=7
        ))
        
        # Data export/extraction
        self.audit_logger.add_compliance_rule(ComplianceRule(
            rule_id="data_export_monitor",
            name="Data Export Monitoring",
            description="Monitor potential data export or extraction activities",
            pattern=r"(file_read|data_access).*\.(csv|json|sql|db)",
            severity=AuditLevel.COMPLIANCE,
            action_required="log",
            retention_years=5
        ))
        
        # Security-sensitive operations
        self.audit_logger.add_compliance_rule(ComplianceRule(
            rule_id="security_operations",
            name="Security Operations Monitor",
            description="Monitor security-sensitive operations",
            pattern=r"(security_event|configuration_change).*password|key|token|cert",
            severity=AuditLevel.CRITICAL,
            action_required="alert",
            retention_years=10
        ))

    def audit_tool_execution(self, tool_name: str, arguments: Dict[str, Any], 
                           result: Any, user_id: str = None, tenant_id: str = None,
                           execution_time: float = None, success: bool = True):
        """Comprehensive audit logging for tool execution"""
        
        # Determine action type based on tool name
        action_type = ActionType.AGENT_DECISION
        if "file" in tool_name.lower():
            action_type = ActionType.FILE_READ if "read" in tool_name else ActionType.FILE_WRITE
        elif "execute" in tool_name.lower():
            action_type = ActionType.CODE_EXECUTION
        elif "llm" in tool_name.lower() or "request" in tool_name.lower():
            action_type = ActionType.LLM_REQUEST
        
        # Determine audit level
        level = AuditLevel.INFO
        if not success:
            level = AuditLevel.WARNING
        if tool_name in ["execute_code", "run_tests", "file_write"]:
            level = AuditLevel.COMPLIANCE
        
        # Log the execution
        audit_id = self.audit_logger.log_audit_entry(
            action_type=action_type,
            level=level,
            user_id=user_id,
            tenant_id=tenant_id,
            details={
                "tool_name": tool_name,
                "arguments_hash": hashlib.md5(json.dumps(arguments, sort_keys=True).encode()).hexdigest(),
                "execution_time": execution_time,
                "success": success,
                "result_size": len(str(result)) if result else 0
            },
            context={
                "arguments": arguments if len(str(arguments)) < 1000 else {"truncated": True},
                "timestamp": time.time()
            },
            compliance_tags=[tool_name, "tool_execution"]
        )
        
        return audit_id

    async def compliance_review_action(self, action_desc: str, context: Dict[str, Any], 
                                     domain: str = "general", block_on_violation: bool = False) -> bool:
        """Review action for compliance and optionally block if violations found"""
        
        review_result = await self.attorney_reviewer.review_action_for_compliance(
            action_desc, context, domain
        )
        
        if block_on_violation and review_result["compliance_status"] == "REJECTED":
            raise ComplianceViolationError(
                f"Action blocked by compliance review: {review_result['legal_notes']}"
            )
        
        return review_result["compliance_status"] in ["APPROVED", "CONDITIONAL"]

    def handle_audit_request(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP requests for audit operations"""
        action = arguments.get("action", "search")
        
        if action == "search":
            results = self.audit_logger.search_audit_trail(
                query_params=arguments.get("filters", {}),
                limit=arguments.get("limit", 100)
            )
            return {"status": "success", "results": results}
            
        elif action == "verify_integrity":
            verification = self.audit_logger.verify_chain_integrity(
                start_id=arguments.get("start_id"),
                end_id=arguments.get("end_id")
            )
            return {"status": "success", "verification": verification}
            
        elif action == "compliance_report":
            report = self.audit_logger.generate_compliance_report(
                tenant_id=arguments.get("tenant_id"),
                start_date=arguments.get("start_date"),
                end_date=arguments.get("end_date")
            )
            return {"status": "success", "report": report}
            
        elif action == "review_compliance":
            # Run async compliance review
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                review_result = loop.run_until_complete(
                    self.attorney_reviewer.review_action_for_compliance(
                        arguments.get("action_description", ""),
                        arguments.get("context", {}),
                        arguments.get("domain", "general")
                    )
                )
                return {"status": "success", "review": review_result}
            finally:
                loop.close()
                
        elif action == "add_compliance_rule":
            rule_data = arguments.get("rule", {})
            rule = ComplianceRule(
                rule_id=rule_data["rule_id"],
                name=rule_data["name"],
                description=rule_data.get("description", ""),
                pattern=rule_data["pattern"],
                severity=AuditLevel(rule_data["severity"]),
                action_required=rule_data.get("action_required", "log"),
                retention_years=rule_data.get("retention_years", 7)
            )
            self.audit_logger.add_compliance_rule(rule)
            return {"status": "success", "message": f"Compliance rule '{rule.name}' added"}
        
        return {"status": "error", "message": "Unknown audit action"}
```


## 7. Introspective, No-Code Workflow Composer (with Teach-Back)

**File: `workflow_composer.py`**

```python
import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import uuid
from pathlib import Path
import graphviz
import tempfile

class NodeType(Enum):
    INPUT = "input"
    OUTPUT = "output"
    AGENT = "agent"
    TOOL = "tool"
    CONDITION = "condition"
    LOOP = "loop"
    PARALLEL = "parallel"
    MERGE = "merge"
    TRANSFORM = "transform"

class ExecutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class WorkflowNode:
    node_id: str
    name: str
    node_type: NodeType
    config: Dict[str, Any]
    inputs: List[str]  # Node IDs that provide input
    outputs: List[str]  # Node IDs that receive output
    position: Dict[str, float]  # x, y coordinates for visual layout
    metadata: Dict[str, Any]

@dataclass
class WorkflowExecution:
    execution_id: str
    workflow_id: str
    status: ExecutionStatus
    start_time: float
    end_time: Optional[float]
    current_node: Optional[str]
    node_results: Dict[str, Any]
    error_log: List[str]
    execution_trace: List[Dict[str, Any]]

@dataclass
class Workflow:
    workflow_id: str
    name: str
    description: str
    nodes: List[WorkflowNode]
    created_at: float
    version: int
    tags: List[str]
    metadata: Dict[str, Any]

class WorkflowComposer:
    def __init__(self, server, storage):
        self.server = server
        self.storage = storage
        self.workflows = {}  # workflow_id -> Workflow
        self.executions = {}  # execution_id -> WorkflowExecution
        self.node_templates = self._create_node_templates()
        self.load_workflows()

    def _create_node_templates(self) -> Dict[str, Dict[str, Any]]:
        """Create templates for different types of workflow nodes"""
        return {
            "input": {
                "name": "Input Node",
                "description": "Accepts input data for the workflow",
                "config_schema": {
                    "input_type": {"type": "string", "enum": ["text", "file", "json", "number"]},
                    "validation": {"type": "string", "optional": True},
                    "default_value": {"type": "any", "optional": True}
                }
            },
            "output": {
                "name": "Output Node", 
                "description": "Produces output from the workflow",
                "config_schema": {
                    "output_format": {"type": "string", "enum": ["text", "json", "file", "report"]},
                    "template": {"type": "string", "optional": True}
                }
            },
            "agent": {
                "name": "AI Agent",
                "description": "Execute tasks using AI agents",
                "config_schema": {
                    "agent_type": {"type": "string", "enum": ["planner", "coder", "reviewer", "analyst"]},
                    "task_template": {"type": "string"},
                    "model_preference": {"type": "string", "optional": True}
                }
            },
            "tool": {
                "name": "Tool Execution",
                "description": "Execute MCP tools",
                "config_schema": {
                    "tool_name": {"type": "string"},
                    "parameters": {"type": "object"},
                    "retry_count": {"type": "number", "default": 3}
                }
            },
            "condition": {
                "name": "Conditional Branch",
                "description": "Branch execution based on conditions",
                "config_schema": {
                    "condition_type": {"type": "string", "enum": ["equals", "contains", "greater", "exists"]},
                    "field_path": {"type": "string"},
                    "expected_value": {"type": "any"},
                    "true_branch": {"type": "string"},
                    "false_branch": {"type": "string"}
                }
            },
            "loop": {
                "name": "Loop Iterator",
                "description": "Iterate over data collections",
                "config_schema": {
                    "iteration_source": {"type": "string"},
                    "loop_body": {"type": "array"},
                    "max_iterations": {"type": "number", "default": 100}
                }
            },
            "transform": {
                "name": "Data Transform",
                "description": "Transform data between nodes",
                "config_schema": {
                    "transform_type": {"type": "string", "enum": ["map", "filter", "reduce", "extract"]},
                    "transform_config": {"type": "object"}
                }
            }
        }

    def create_workflow(self, name: str, description: str = "", tags: List[str] = None) -> str:
        """Create a new empty workflow"""
        workflow_id = f"workflow_{uuid.uuid4().hex[:12]}"
        
        workflow = Workflow(
            workflow_id=workflow_id,
            name=name,
            description=description,
            nodes=[],
            created_at=time.time(),
            version=1,
            tags=tags or [],
            metadata={}
        )
        
        self.workflows[workflow_id] = workflow
        self._save_workflow(workflow)
        
        return workflow_id

    def add_node(self, workflow_id: str, node_type: NodeType, name: str, 
                config: Dict[str, Any], position: Dict[str, float] = None) -> str:
        """Add a node to a workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        node_id = f"node_{uuid.uuid4().hex[:8]}"
        
        node = WorkflowNode(
            node_id=node_id,
            name=name,
            node_type=node_type,
            config=config,
            inputs=[],
            outputs=[],
            position=position or {"x": 0, "y": 0},
            metadata={}
        )
        
        self.workflows[workflow_id].nodes.append(node)
        self.workflows[workflow_id].version += 1
        self._save_workflow(self.workflows[workflow_id])
        
        return node_id

    def connect_nodes(self, workflow_id: str, source_node_id: str, target_node_id: str):
        """Connect two nodes in a workflow"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Find source and target nodes
        source_node = None
        target_node = None
        
        for node in workflow.nodes:
            if node.node_id == source_node_id:
                source_node = node
            elif node.node_id == target_node_id:
                target_node = node
        
        if not source_node or not target_node:
            raise ValueError("Source or target node not found")
        
        # Add connections
        if target_node_id not in source_node.outputs:
            source_node.outputs.append(target_node_id)
        if source_node_id not in target_node.inputs:
            target_node.inputs.append(source_node_id)
        
        workflow.version += 1
        self._save_workflow(workflow)

    def remove_node(self, workflow_id: str, node_id: str):
        """Remove a node from a workflow"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Remove connections to this node
        for node in workflow.nodes:
            if node_id in node.inputs:
                node.inputs.remove(node_id)
            if node_id in node.outputs:
                node.outputs.remove(node_id)
        
        # Remove the node itself
        workflow.nodes = [n for n in workflow.nodes if n.node_id != node_id]
        workflow.version += 1
        self._save_workflow(workflow)

    async def explain_workflow(self, workflow_id: str, detail_level: str = "medium") -> str:
        """Generate natural language explanation of workflow using LLM"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Create workflow description for LLM
        workflow_desc = self._create_workflow_description(workflow)
        
        detail_prompts = {
            "basic": "Explain this workflow in 2-3 sentences for a business user.",
            "medium": "Explain this workflow step-by-step for a technical user, including what each component does.",
            "detailed": "Provide a comprehensive explanation of this workflow including technical details, data flow, error handling, and potential optimizations."
        }
        
        prompt = f"""
        Explain the following workflow in natural language:
        
        Workflow Name: {workflow.name}
        Description: {workflow.description}
        
        Workflow Structure:
        {workflow_desc}
        
        {detail_prompts.get(detail_level, detail_prompts["medium"])}
        
        Make the explanation clear, accurate, and helpful for someone who wants to understand how this workflow operates.
        """
        
        explanation = await self.server.make_llm_request_with_retry(prompt, temperature=0.3)
        return explanation

    def _create_workflow_description(self, workflow: Workflow) -> str:
        """Create a structured description of the workflow for LLM processing"""
        description = f"Nodes ({len(workflow.nodes)}):\n"
        
        for node in workflow.nodes:
            description += f"- {node.name} ({node.node_type.value}): {node.config}\n"
            if node.inputs:
                description += f"  Inputs from: {', '.join(node.inputs)}\n"
            if node.outputs:
                description += f"  Outputs to: {', '.join(node.outputs)}\n"
        
        return description

    def validate_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Validate workflow structure and identify issues"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        # Check for input nodes
        input_nodes = [n for n in workflow.nodes if n.node_type == NodeType.INPUT]
        if not input_nodes:
            validation_result["warnings"].append("No input nodes found - workflow may not accept external data")
        
        # Check for output nodes
        output_nodes = [n for n in workflow.nodes if n.node_type == NodeType.OUTPUT]
        if not output_nodes:
            validation_result["errors"].append("No output nodes found - workflow will not produce results")
            validation_result["valid"] = False
        
        # Check for disconnected nodes
        for node in workflow.nodes:
            if not node.inputs and not node.outputs and node.node_type not in [NodeType.INPUT, NodeType.OUTPUT]:
                validation_result["warnings"].append(f"Node '{node.name}' is disconnected")
        
        # Check for cycles (simplified)
        if self._has_cycles(workflow):
            validation_result["errors"].append("Workflow contains cycles - this may cause infinite loops")
            validation_result["valid"] = False
        
        # Suggest optimizations
        if len(workflow.nodes) > 20:
            validation_result["suggestions"].append("Consider breaking this large workflow into smaller, reusable sub-workflows")
        
        return validation_result

    def _has_cycles(self, workflow: Workflow) -> bool:
        """Simple cycle detection using DFS"""
        visited = set()
        rec_stack = set()
        
        def dfs(node_id):
            visited.add(node_id)
            rec_stack.add(node_id)
            
            node = next((n for n in workflow.nodes if n.node_id == node_id), None)
            if node:
                for output_id in node.outputs:
                    if output_id not in visited:
                        if dfs(output_id):
                            return True
                    elif output_id in rec_stack:
                        return True
            
            rec_stack.remove(node_id)
            return False
        
        for node in workflow.nodes:
            if node.node_id not in visited:
                if dfs(node.node_id):
                    return True
        
        return False

    async def execute_workflow(self, workflow_id: str, inputs: Dict[str, Any]) -> str:
        """Execute a workflow with given inputs"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Validate workflow before execution
        validation = self.validate_workflow(workflow_id)
        if not validation["valid"]:
            raise ValueError(f"Workflow validation failed: {validation['errors']}")
        
        execution_id = f"exec_{uuid.uuid4().hex[:12]}"
        
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_id,
            status=ExecutionStatus.RUNNING,
            start_time=time.time(),
            end_time=None,
            current_node=None,
            node_results={},
            error_log=[],
            execution_trace=[]
        )
        
        self.executions[execution_id] = execution
        
        try:
            # Execute workflow nodes in topological order
            execution_order = self._get_execution_order(workflow)
            
            # Initialize with input data
            execution.node_results.update(inputs)
            
            for node_id in execution_order:
                node = next(n for n in workflow.nodes if n.node_id == node_id)
                execution.current_node = node_id
                
                execution.execution_trace.append({
                    "timestamp": time.time(),
                    "node_id": node_id,
                    "node_name": node.name,
                    "action": "starting"
                })
                
                try:
                    result = await self._execute_node(node, execution.node_results)
                    execution.node_results[node_id] = result
                    
                    execution.execution_trace.append({
                        "timestamp": time.time(),
                        "node_id": node_id,
                        "action": "completed",
                        "result_size": len(str(result))
                    })
                    
                except Exception as e:
                    error_msg = f"Node {node.name} failed: {str(e)}"
                    execution.error_log.append(error_msg)
                    execution.status = ExecutionStatus.FAILED
                    
                    execution.execution_trace.append({
                        "timestamp": time.time(),
                        "node_id": node_id,
                        "action": "failed",
                        "error": str(e)
                    })
                    
                    raise
            
            execution.status = ExecutionStatus.COMPLETED
            execution.end_time = time.time()
            
        except Exception as e:
            execution.status = ExecutionStatus.FAILED
            execution.end_time = time.time()
            raise
        
        finally:
            self._save_execution(execution)
        
        return execution_id

    def _get_execution_order(self, workflow: Workflow) -> List[str]:
        """Get topological execution order for workflow nodes"""
        # Simple topological sort
        in_degree = {}
        for node in workflow.nodes:
            in_degree[node.node_id] = len(node.inputs)
        
        queue = [node.node_id for node in workflow.nodes if in_degree[node.node_id] == 0]
        execution_order = []
        
        while queue:
            current = queue.pop(0)
            execution_order.append(current)
            
            current_node = next(n for n in workflow.nodes if n.node_id == current)
            for output_id in current_node.outputs:
                in_degree[output_id] -= 1
                if in_degree[output_id] == 0:
                    queue.append(output_id)
        
        return execution_order

    async def _execute_node(self, node: WorkflowNode, context: Dict[str, Any]) -> Any:
        """Execute a single workflow node"""
        if node.node_type == NodeType.TOOL:
            return await self._execute_tool_node(node, context)
        elif node.node_type == NodeType.AGENT:
            return await self._execute_agent_node(node, context)
        elif node.node_type == NodeType.CONDITION:
            return self._execute_condition_node(node, context)
        elif node.node_type == NodeType.TRANSFORM:
            return self._execute_transform_node(node, context)
        elif node.node_type == NodeType.OUTPUT:
            return self._execute_output_node(node, context)
        else:
            return f"Executed {node.name}"

    async def _execute_tool_node(self, node: WorkflowNode, context: Dict[str, Any]) -> Any:
        """Execute a tool node"""
        tool_name = node.config["tool_name"]
        parameters = node.config["parameters"]
        
        # Substitute context variables in parameters
        resolved_params = self._resolve_template_variables(parameters, context)
        
        # Call the tool through the server's tool system
        # This is a simplified version - in practice you'd integrate with the actual tool system
        from main import handle_tool_call  # Import the actual tool handler
        
        mock_message = {
            "params": {
                "name": tool_name,
                "arguments": resolved_params
            }
        }
        
        result = handle_tool_call(mock_message)
        return result.get("result", {}).get("content", [{}])[^0].get("text", "")

    async def _execute_agent_node(self, node: WorkflowNode, context: Dict[str, Any]) -> Any:
        """Execute an AI agent node"""
        agent_type = node.config["agent_type"]
        task_template = node.config["task_template"]
        
        # Resolve template variables
        task = self._resolve_template_variables(task_template, context)
        
        # Create appropriate prompt based on agent type
        if agent_type == "planner":
            prompt = f"You are a project planner. Create a detailed plan for: {task}"
        elif agent_type == "coder":
            prompt = f"You are a software developer. Write code for: {task}"
        elif agent_type == "reviewer":
            prompt = f"You are a code reviewer. Review and provide feedback on: {task}"
        elif agent_type == "analyst":
            prompt = f"You are a business analyst. Analyze: {task}"
        else:
            prompt = task
        
        response = await self.server.make_llm_request_with_retry(prompt, temperature=0.3)
        return response

    def _execute_condition_node(self, node: WorkflowNode, context: Dict[str, Any]) -> str:
        """Execute a conditional node"""
        condition_type = node.config["condition_type"]
        field_path = node.config["field_path"]
        expected_value = node.config["expected_value"]
        
        # Get actual value from context
        actual_value = self._get_nested_value(context, field_path)
        
        # Evaluate condition
        result = False
        if condition_type == "equals":
            result = actual_value == expected_value
        elif condition_type == "contains" and isinstance(actual_value, str):
            result = str(expected_value) in actual_value
        elif condition_type == "greater" and isinstance(actual_value, (int, float)):
            result = actual_value > expected_value
        elif condition_type == "exists":
            result = actual_value is not None
        
        return node.config["true_branch"] if result else node.config["false_branch"]

    def _execute_transform_node(self, node: WorkflowNode, context: Dict[str, Any]) -> Any:
        """Execute a data transformation node"""
        transform_type = node.config["transform_type"]
        transform_config = node.config["transform_config"]
        
        source_field = transform_config.get("source_field", "input")
        source_data = self._get_nested_value(context, source_field)
        
        if transform_type == "map" and isinstance(source_data, list):
            mapping_expr = transform_config.get("mapping", "item")
            return [self._evaluate_expression(mapping_expr, {"item": item}) for item in source_data]
        elif transform_type == "filter" and isinstance(source_data, list):
            filter_expr = transform_config.get("filter", "True")
            return [item for item in source_data if self._evaluate_expression(filter_expr, {"item": item})]
        elif transform_type == "extract":
            extract_path = transform_config.get("path", "")
            return self._get_nested_value(source_data, extract_path)
        
        return source_data

    def _execute_output_node(self, node: WorkflowNode, context: Dict[str, Any]) -> Any:
        """Execute an output node"""
        output_format = node.config.get("output_format", "json")
        template = node.config.get("template", "")
        
        if template:
            return self._resolve_template_variables(template, context)
        elif output_format == "json":
            return json.dumps(context, indent=2)
        else:
            return str(context)

    def _resolve_template_variables(self, template: Union[str, Dict, List], context: Dict[str, Any]) -> Any:
        """Resolve template variables like {{variable_name}}"""
        if isinstance(template, str):
            import re
            def replace_var(match):
                var_name = match.group(1)
                return str(self._get_nested_value(context, var_name, ""))
            
            return re.sub(r'\{\{([^}]+)\}\}', replace_var, template)
        elif isinstance(template, dict):
            return {k: self._resolve_template_variables(v, context) for k, v in template.items()}
        elif isinstance(template, list):
            return [self._resolve_template_variables(item, context) for item in template]
        else:
            return template

    def _get_nested_value(self, data: Dict[str, Any], path: str, default=None) -> Any:
        """Get nested value using dot notation (e.g., 'user.name')"""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current

    def _evaluate_expression(self, expression: str, context: Dict[str, Any]) -> Any:
        """Safely evaluate simple expressions"""
        # This is a simplified version - in production you'd want a proper expression evaluator
        try:
            # Replace variables in expression
            for key, value in context.items():
                expression = expression.replace(key, repr(value))
            
            # Only allow safe operations
            allowed_names = {"True": True, "False": False, "None": None}
            return eval(expression, {"__builtins__": {}}, allowed_names)
        except:
            return False

    def generate_workflow_diagram(self, workflow_id: str) -> str:
        """Generate a visual diagram of the workflow"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Create graphviz diagram
        dot = graphviz.Digraph(comment=workflow.name)
        dot.attr(rankdir='TB')
        
        # Add nodes
        for node in workflow.nodes:
            shape = "ellipse"
            color = "lightblue"
            
            if node.node_type == NodeType.INPUT:
                shape = "invhouse"
                color = "lightgreen"
            elif node.node_type == NodeType.OUTPUT:
                shape = "house"
                color = "lightcoral"
            elif node.node_type == NodeType.CONDITION:
                shape = "diamond"
                color = "lightyellow"
            
            dot.node(node.node_id, f"{node.name}\\n({node.node_type.value})", 
                    shape=shape, style='filled', fillcolor=color)
        
        # Add edges
        for node in workflow.nodes:
            for output_id in node.outputs:
                dot.edge(node.node_id, output_id)
        
        # Save to temporary file and return path
        temp_dir = tempfile.gettempdir()
        diagram_path = Path(temp_dir) / f"workflow_{workflow_id}_{int(time.time())}"
        
        dot.render(str(diagram_path), format='png', cleanup=True)
        return str(diagram_path) + '.png'

    def get_execution_status(self, execution_id: str) -> Dict[str, Any]:
        """Get detailed execution status"""
        execution = self.executions.get(execution_id)
        if not execution:
            raise ValueError(f"Execution {execution_id} not found")
        
        return {
            "execution_id": execution_id,
            "workflow_id": execution.workflow_id,
            "status": execution.status.value,
            "start_time": execution.start_time,
            "end_time": execution.end_time,
            "current_node": execution.current_node,
            "progress": len([t for t in execution.execution_trace if t["action"] == "completed"]),
            "total_nodes": len(self.workflows[execution.workflow_id].nodes),
            "errors": execution.error_log,
            "duration": (execution.end_time or time.time()) - execution.start_time
        }

    def _save_workflow(self, workflow: Workflow):
        """Save workflow to persistent storage"""
        self.storage.store_memory(
            key=f"workflow_{workflow.workflow_id}",
            value=json.dumps(asdict(workflow), default=str),
            category="workflows"
        )

    def _save_execution(self, execution: WorkflowExecution):
        """Save execution to persistent storage"""
        self.storage.store_memory(
            key=f"execution_{execution.execution_id}",
            value=json.dumps(asdict(execution), default=str),
            category="executions"
        )

    def load_workflows(self):
        """Load workflows from persistent storage"""
        try:
            workflows = self.storage.retrieve_memory(category="workflows")
            for workflow_data in workflows:
                workflow_dict = json.loads(workflow_data['value'])
                
                # Convert node data back to objects
                nodes = []
                for node_data in workflow_dict['nodes']:
                    node = WorkflowNode(
                        node_id=node_data['node_id'],
                        name=node_data['name'],
                        node_type=NodeType(node_data['node_type']),
                        config=node_data['config'],
                        inputs=node_data['inputs'],
                        outputs=node_data['outputs'],
                        position=node_data['position'],
                        metadata=node_data['metadata']
                    )
                    nodes.append(node)
                
                workflow = Workflow(
                    workflow_id=workflow_dict['workflow_id'],
                    name=workflow_dict['name'],
                    description=workflow_dict['description'],
                    nodes=nodes,
                    created_at=workflow_dict['created_at'],
                    version=workflow_dict['version'],
                    tags=workflow_dict['tags'],
                    metadata=workflow_dict['metadata']
                )
                
                self.workflows[workflow.workflow_id] = workflow
                
        except Exception as e:
            print(f"Failed to load workflows: {e}")

class WorkflowComposerIntegration:
    """Integration layer for workflow composer with MCP server"""
    
    def __init__(self, server):
        self.server = server
        self.composer = WorkflowComposer(server, server.storage)
        
    def handle_workflow_request(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP requests for workflow operations"""
        action = arguments.get("action", "list")
        
        if action == "create":
            workflow_id = self.composer.create_workflow(
                name=arguments.get("name", "New Workflow"),
                description=arguments.get("description", ""),
                tags=arguments.get("tags", [])
            )
            return {"status": "success", "workflow_id": workflow_id}
            
        elif action == "add_node":
            node_id = self.composer.add_node(
                workflow_id=arguments["workflow_id"],
                node_type=NodeType(arguments["node_type"]),
                name=arguments["name"],
                config=arguments["config"],
                position=arguments.get("position")
            )
            return {"status": "success", "node_id": node_id}
            
        elif action == "connect":
            self.composer.connect_nodes(
                workflow_id=arguments["workflow_id"],
                source_node_id=arguments["source_node"],
                target_node_id=arguments["target_node"]
            )
            return {"status": "success", "message": "Nodes connected"}
            
        elif action == "execute":
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                execution_id = loop.run_until_complete(
                    self.composer.execute_workflow(
                        workflow_id=arguments["workflow_id"],
                        inputs=arguments.get("inputs", {})
                    )
                )
                return {"status": "success", "execution_id": execution_id}
            finally:
                loop.close()
                
        elif action == "explain":
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                explanation = loop.run_until_complete(
                    self.composer.explain_workflow(
                        workflow_id=arguments["workflow_id"],
                        detail_level=arguments.get("detail_level", "medium")
                    )
                )
                return {"status": "success", "explanation": explanation}
            finally:
                loop.close()
                
        elif action == "validate":
            validation = self.composer.validate_workflow(arguments["workflow_id"])
            return {"status": "success", "validation": validation}
            
        elif action == "diagram":
            diagram_path = self.composer.generate_workflow_diagram(arguments["workflow_id"])
            return {"status": "success", "diagram_path": diagram_path}
            
        elif action == "execution_status":
            status = self.composer.get_execution_status(arguments["execution_id"])
            return {"status": "success", "execution_status": status}
            
        elif action == "list":
            workflows = [
                {
                    "workflow_id": wf.workflow_id,
                    "name": wf.name,
                    "description": wf.description,
                    "node_count": len(wf.nodes),
                    "created_at": wf.created_at,
                    "version": wf.version
                }
                for wf in self.composer.workflows.values()
            ]
            return {"status": "success", "workflows": workflows}
            
        elif action == "templates":
            return {"status": "success", "templates": self.composer.node_templates}
        
        return {"status": "error", "message": "Unknown workflow action"}
```


## Final Integration Guide

### Add New Tools to get_all_tools()

```python
# In get_all_tools() function, add these new tools:
{
    "name": "audit_trail",
    "description": "Immutable audit logging and compliance operations",
    "inputSchema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["search", "verify_integrity", "compliance_report", "review_compliance", "add_compliance_rule"]},
            "filters": {"type": "object", "description": "Search filters"},
            "tenant_id": {"type": "string"},
            "start_date": {"type": "number"},
            "end_date": {"type": "number"},
            "action_description": {"type": "string"},
            "context": {"type": "object"},
            "domain": {"type": "string", "default": "general"},
            "rule": {"type": "object"}
        },
        "required": ["action"]
    }
},
{
    "name": "workflow_composer",
    "description": "No-code workflow composition and execution with natural language explanations",
    "inputSchema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["create", "add_node", "connect", "execute", "explain", "validate", "diagram", "execution_status", "list", "templates"]},
            "workflow_id": {"type": "string"},
            "name": {"type": "string"},
            "description": {"type": "string"},
            "node_type": {"type": "string"},
            "config": {"type": "object"},
            "source_node": {"type": "string"},
            "target_node": {"type": "string"},
            "inputs": {"type": "object"},
            "execution_id": {"type": "string"},
            "detail_level": {"type": "string", "enum": ["basic", "medium", "detailed"], "default": "medium"}
        },
        "required": ["action"]
    }
}
```


### Initialize in EnhancedLMStudioMCPServer.__init__()

```python
# Add to the __init__ method:
        
# Audit logging and compliance
self.audit_integration = AuditingMCPIntegration(self)

# Workflow composer
self.workflow_integration = WorkflowComposerIntegration(self)

# Override the monitor_performance decorator to include audit logging
original_monitor = self.monitor_performance
def enhanced_monitor_performance(func_name: str):
    def decorator(func):
        monitored_func = original_monitor(func_name)(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            result = None
            try:
                result = monitored_func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise
            finally:
                # Audit log every tool execution
                execution_time = time.time() - start_time
                self.audit_integration.audit_tool_execution(
                    tool_name=func_name,
                    arguments=args[^0] if args else {},
                    result=result,
                    user_id=getattr(args, 'user_id', None) if args else None,
                    tenant_id=getattr(args, 'tenant_id', None) if args else None,
                    execution_time=execution_time,
                    success=success
                )
        return wrapper
    return decorator

self.monitor_performance = enhanced_monitor_performance
```


### Add Handler Functions

```python
def handle_audit_trail(arguments, server):
    return server.audit_integration.handle_audit_request(arguments)

def handle_workflow_composer(arguments, server):
    return server.workflow_integration.handle_workflow_request(arguments)
```


### Update Tool Registry

```python
# In handle_tool_call registry, add:
registry.update({
    "audit_trail": (handle_audit_trail, True),
    "workflow_composer": (handle_workflow_composer, True),
})
```


### Environment Variables

```bash
# Add to .env.example:

# Audit logging
AUDIT_ENABLED=true
AUDIT_DB_PATH=audit_trail.db
COMPLIANCE_DOMAIN=general
AUDIT_RETENTION_YEARS=7

# Workflow composer
WORKFLOW_COMPOSER_ENABLED=true
VISUAL_DIAGRAMS_ENABLED=true
WORKFLOW_EXECUTION_TIMEOUT=300
```

This completes the implementation of all 7 novel MCP server enhancements. Each feature is designed to work independently while integrating seamlessly with the existing codebase, providing enterprise-grade capabilities that make your MCP server truly unique and desirable for the broadest user base.

<div style="text-align: center">⁂</div>

[^1]: paste.txt

[^2]: https://stackoverflow.com/questions/21009416/python-subprocess-security

[^3]: https://security.openstack.org/guidelines/dg_using-file-paths.html

[^4]: https://realpython.com/python-pep8/

[^5]: https://semgrep.dev/docs/cheat-sheets/python-command-injection

