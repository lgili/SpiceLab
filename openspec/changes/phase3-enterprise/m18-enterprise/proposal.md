# M18: Enterprise Features

**Status:** Proposed
**Priority:** 🟡 MEDIUM
**Estimated Duration:** 10-12 weeks
**Dependencies:** M10 (I/O optimization), M15 (distributed), M16 (compliance), All previous milestones

## Problem Statement

SpiceLab lacks enterprise-grade features required for commercial deployment in organizations with security, compliance, collaboration, and resource management requirements. Features like design versioning, audit trails, access control, and license management are essential for enterprise adoption but currently absent.

### Current Gaps
- ❌ No design versioning system (Git-like diffs for circuits)
- ❌ No audit logs (simulation history, parameter changes)
- ❌ No role-based access control (RBAC)
- ❌ No simulation quotas or resource limits
- ❌ No team collaboration features (shared cache, design sharing)
- ❌ No license management (seat licenses, floating licenses)
- ❌ No SSO integration (SAML, OAuth, LDAP)
- ❌ No enterprise security audit compliance

### Impact
- **Enterprise Sales:** Cannot sell to large organizations
- **Security:** No access control or audit trails
- **Collaboration:** Teams cannot work together effectively
- **Scalability:** No resource management or quotas

## Objectives

1. **Design versioning** - Git-like version control for circuits with diffs
2. **Auditability** - Complete simulation history and parameter tracking
3. **Role-based access control** - User roles and permissions
4. **Simulation quotas** - Resource limits per user/team
5. **Team collaboration** - Shared cache, design libraries, comments
6. **License management** - Seat licenses, floating licenses, usage tracking
7. **SSO integration** - SAML 2.0, OAuth 2.0, LDAP authentication
8. **Target:** Enterprise edition beta, SOC 2 Type II audit preparation

## Technical Design

### 1. Design Versioning System

```python
# spicelab/enterprise/versioning.py
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import hashlib
import json

@dataclass
class CircuitVersion:
    """Version of a circuit design."""
    version_id: str  # SHA-256 hash of circuit definition
    circuit_name: str
    author: str
    timestamp: datetime
    message: str  # Commit message
    parent_version: str | None = None  # For history chain
    tags: list[str] = None  # e.g., ["v1.0", "release"]

class DesignVersionControl:
    """Git-like version control for circuits."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.versions_db = repo_path / ".spicelab" / "versions.db"
        self._init_repo()

    def _init_repo(self):
        """Initialize version control repository."""
        (self.repo_path / ".spicelab").mkdir(exist_ok=True)

    def commit(
        self,
        circuit: 'Circuit',
        message: str,
        author: str
    ) -> CircuitVersion:
        """Save a new version of the circuit."""

        # Serialize circuit to canonical JSON
        circuit_json = circuit.to_json(sort_keys=True, indent=None)

        # Calculate version ID (hash of content)
        version_id = hashlib.sha256(circuit_json.encode()).hexdigest()

        # Get parent version (current HEAD)
        parent = self._get_head()

        # Create version object
        version = CircuitVersion(
            version_id=version_id,
            circuit_name=circuit.name,
            author=author,
            timestamp=datetime.now(),
            message=message,
            parent_version=parent.version_id if parent else None
        )

        # Save circuit snapshot
        version_path = self.repo_path / ".spicelab" / "versions" / version_id
        version_path.mkdir(parents=True, exist_ok=True)
        (version_path / "circuit.json").write_text(circuit_json)
        (version_path / "metadata.json").write_text(json.dumps(version.__dict__, default=str))

        # Update HEAD
        self._set_head(version)

        return version

    def diff(self, version_a: str, version_b: str) -> dict:
        """Show differences between two versions."""

        circuit_a = self._load_version(version_a)
        circuit_b = self._load_version(version_b)

        diff = {
            "components_added": [],
            "components_removed": [],
            "components_modified": [],
            "parameters_changed": {}
        }

        # Compare components
        refs_a = {c.ref for c in circuit_a.components}
        refs_b = {c.ref for c in circuit_b.components}

        diff["components_added"] = list(refs_b - refs_a)
        diff["components_removed"] = list(refs_a - refs_b)

        # Check modified components
        for ref in refs_a & refs_b:
            comp_a = circuit_a.get_component(ref)
            comp_b = circuit_b.get_component(ref)

            if comp_a != comp_b:
                diff["components_modified"].append({
                    "ref": ref,
                    "changes": self._component_diff(comp_a, comp_b)
                })

        return diff

    def log(self, max_count: int = 10) -> list[CircuitVersion]:
        """Show version history."""
        history = []
        current = self._get_head()

        while current and len(history) < max_count:
            history.append(current)
            if current.parent_version:
                current = self._load_version_metadata(current.parent_version)
            else:
                break

        return history

    def checkout(self, version_id: str) -> 'Circuit':
        """Load a specific version."""
        return self._load_version(version_id)

    def tag(self, version_id: str, tag_name: str):
        """Tag a version (e.g., "v1.0", "release")."""
        ...
```

### 2. Audit Logging

```python
# spicelab/enterprise/audit.py
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

class AuditEventType(Enum):
    """Types of auditable events."""
    SIMULATION_START = "simulation_start"
    SIMULATION_COMPLETE = "simulation_complete"
    SIMULATION_FAILED = "simulation_failed"
    CIRCUIT_CREATED = "circuit_created"
    CIRCUIT_MODIFIED = "circuit_modified"
    CIRCUIT_DELETED = "circuit_deleted"
    PARAMETER_CHANGED = "parameter_changed"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    PERMISSION_CHANGED = "permission_changed"

@dataclass
class AuditEvent:
    """Audit log entry."""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: str
    resource: str  # Circuit name, job ID, etc.
    action: str
    details: dict
    ip_address: str | None = None

class AuditLogger:
    """Centralized audit logging."""

    def __init__(self, database_path: Path):
        self.db_path = database_path
        self.logger = logging.getLogger("spicelab.audit")

    def log_event(self, event: AuditEvent):
        """Log an audit event."""
        # Write to database
        self._write_to_db(event)

        # Also log to file for compliance
        self.logger.info(
            f"{event.timestamp.isoformat()} | {event.user_id} | "
            f"{event.event_type.value} | {event.resource} | {event.action}"
        )

    def log_simulation(
        self,
        user_id: str,
        circuit_name: str,
        simulation_id: str,
        parameters: dict
    ):
        """Log simulation execution."""
        event = AuditEvent(
            event_id=f"sim_{simulation_id}",
            event_type=AuditEventType.SIMULATION_START,
            timestamp=datetime.now(),
            user_id=user_id,
            resource=circuit_name,
            action="run_simulation",
            details={"parameters": parameters, "simulation_id": simulation_id}
        )
        self.log_event(event)

    def query_logs(
        self,
        user_id: str | None = None,
        event_type: AuditEventType | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None
    ) -> list[AuditEvent]:
        """Query audit logs with filters."""
        # Query database with filters
        ...

    def generate_audit_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> str:
        """Generate compliance audit report."""
        events = self.query_logs(start_date=start_date, end_date=end_date)

        report = f"""
# Audit Report
**Period:** {start_date.date()} to {end_date.date()}
**Total Events:** {len(events)}

## Summary by Event Type
"""
        # Aggregate by event type
        from collections import Counter
        event_counts = Counter(e.event_type.value for e in events)

        for event_type, count in event_counts.most_common():
            report += f"- {event_type}: {count}\n"

        report += "\n## Detailed Events\n"
        for event in events[:100]:  # Limit to first 100
            report += f"- {event.timestamp} | {event.user_id} | {event.action}\n"

        return report
```

### 3. Role-Based Access Control (RBAC)

```python
# spicelab/enterprise/rbac.py
from enum import Enum
from dataclasses import dataclass

class Role(Enum):
    """User roles."""
    ADMIN = "admin"
    DESIGNER = "designer"
    VIEWER = "viewer"
    AUDITOR = "auditor"

class Permission(Enum):
    """Permissions."""
    READ_CIRCUIT = "read_circuit"
    WRITE_CIRCUIT = "write_circuit"
    DELETE_CIRCUIT = "delete_circuit"
    RUN_SIMULATION = "run_simulation"
    VIEW_RESULTS = "view_results"
    MANAGE_USERS = "manage_users"
    VIEW_AUDIT_LOGS = "view_audit_logs"

# Role → Permissions mapping
ROLE_PERMISSIONS = {
    Role.ADMIN: [p for p in Permission],  # All permissions
    Role.DESIGNER: [
        Permission.READ_CIRCUIT,
        Permission.WRITE_CIRCUIT,
        Permission.RUN_SIMULATION,
        Permission.VIEW_RESULTS
    ],
    Role.VIEWER: [
        Permission.READ_CIRCUIT,
        Permission.VIEW_RESULTS
    ],
    Role.AUDITOR: [
        Permission.VIEW_AUDIT_LOGS,
        Permission.READ_CIRCUIT,
        Permission.VIEW_RESULTS
    ]
}

@dataclass
class User:
    """User account."""
    user_id: str
    email: str
    role: Role
    enabled: bool = True

class AccessController:
    """RBAC enforcement."""

    def __init__(self):
        self.users: dict[str, User] = {}

    def add_user(self, user: User):
        """Register a user."""
        self.users[user.user_id] = user

    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has permission."""
        user = self.users.get(user_id)
        if not user or not user.enabled:
            return False

        return permission in ROLE_PERMISSIONS[user.role]

    def require_permission(self, user_id: str, permission: Permission):
        """Decorator to enforce permission."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not self.check_permission(user_id, permission):
                    raise PermissionError(
                        f"User {user_id} lacks permission: {permission.value}"
                    )
                return func(*args, **kwargs)
            return wrapper
        return decorator
```

### 4. Simulation Quotas

```python
# spicelab/enterprise/quotas.py
from dataclasses import dataclass

@dataclass
class ResourceQuota:
    """Resource usage quota."""
    user_id: str
    max_simulations_per_day: int = 100
    max_parallel_jobs: int = 10
    max_cpu_hours_per_month: float = 1000.0
    max_storage_gb: float = 100.0

class QuotaManager:
    """Manage and enforce resource quotas."""

    def __init__(self):
        self.quotas: dict[str, ResourceQuota] = {}
        self.usage: dict[str, dict] = {}

    def set_quota(self, user_id: str, quota: ResourceQuota):
        """Set quota for user."""
        self.quotas[user_id] = quota

    def check_quota(self, user_id: str, resource: str, amount: float) -> bool:
        """Check if user is within quota."""
        quota = self.quotas.get(user_id)
        if not quota:
            return True  # No quota = unlimited

        usage = self.usage.get(user_id, {})

        if resource == "simulations_per_day":
            return usage.get(resource, 0) + amount <= quota.max_simulations_per_day
        elif resource == "cpu_hours_per_month":
            return usage.get(resource, 0) + amount <= quota.max_cpu_hours_per_month
        # ... other resources

        return True

    def consume_quota(self, user_id: str, resource: str, amount: float):
        """Consume quota (increment usage)."""
        if user_id not in self.usage:
            self.usage[user_id] = {}

        self.usage[user_id][resource] = self.usage[user_id].get(resource, 0) + amount

    def reset_daily_quotas(self):
        """Reset daily quotas (run via cron)."""
        for user_id in self.usage:
            self.usage[user_id]["simulations_per_day"] = 0
```

### 5. License Management

```python
# spicelab/enterprise/licensing.py
from datetime import datetime, timedelta
import jwt

@dataclass
class License:
    """Software license."""
    license_key: str
    license_type: str  # "seat", "floating", "enterprise"
    max_users: int
    expiration_date: datetime
    features: list[str]  # ["distributed", "pdk", "enterprise"]

class LicenseManager:
    """Manage software licenses."""

    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.active_licenses: dict[str, License] = {}

    def validate_license(self, license_key: str) -> License | None:
        """Validate license key using JWT."""
        try:
            payload = jwt.decode(license_key, self.secret_key, algorithms=["HS256"])

            license = License(
                license_key=license_key,
                license_type=payload["type"],
                max_users=payload["max_users"],
                expiration_date=datetime.fromisoformat(payload["expiration"]),
                features=payload["features"]
            )

            # Check expiration
            if license.expiration_date < datetime.now():
                return None

            return license

        except jwt.InvalidTokenError:
            return None

    def check_feature(self, license_key: str, feature: str) -> bool:
        """Check if license includes feature."""
        license = self.validate_license(license_key)
        if not license:
            return False

        return feature in license.features

    def generate_license(
        self,
        license_type: str,
        max_users: int,
        duration_days: int,
        features: list[str]
    ) -> str:
        """Generate license key (admin only)."""
        payload = {
            "type": license_type,
            "max_users": max_users,
            "expiration": (datetime.now() + timedelta(days=duration_days)).isoformat(),
            "features": features
        }

        license_key = jwt.encode(payload, self.secret_key, algorithm="HS256")
        return license_key
```

### 6. SSO Integration

```python
# spicelab/enterprise/sso.py
from onelogin.saml2.auth import OneLogin_Saml2_Auth

class SSOProvider:
    """Single Sign-On integration."""

    def __init__(self, saml_settings: dict):
        self.saml_settings = saml_settings

    def initiate_sso(self, redirect_url: str) -> str:
        """Initiate SAML SSO flow."""
        auth = OneLogin_Saml2_Auth(request={}, old_settings=self.saml_settings)
        return auth.login(return_to=redirect_url)

    def process_sso_response(self, saml_response: str) -> dict:
        """Process SAML response from IdP."""
        auth = OneLogin_Saml2_Auth(request={}, old_settings=self.saml_settings)
        auth.process_response()

        if not auth.is_authenticated():
            raise ValueError("SAML authentication failed")

        user_data = {
            "user_id": auth.get_nameid(),
            "email": auth.get_attribute("email")[0],
            "name": auth.get_attribute("name")[0],
            "groups": auth.get_attribute("groups")
        }

        return user_data
```

## Implementation Plan

### Phase 1: Versioning (Weeks 1-3)
- [ ] DesignVersionControl class (Git-like)
- [ ] Circuit diff algorithm
- [ ] Version history and tags
- [ ] Merge conflict resolution
- [ ] CLI commands (commit, log, diff, checkout)

### Phase 2: Audit Logging (Weeks 4-5)
- [ ] AuditLogger with database backend
- [ ] Event types and schema
- [ ] Query interface
- [ ] Compliance report generator
- [ ] Integration with all critical operations

### Phase 3: RBAC (Weeks 6-7)
- [ ] Role and permission system
- [ ] AccessController enforcement
- [ ] User management API
- [ ] Permission decorators for API methods

### Phase 4: Quotas & Licensing (Weeks 8-9)
- [ ] QuotaManager with usage tracking
- [ ] LicenseManager with JWT validation
- [ ] Floating license server (optional)
- [ ] Usage dashboard

### Phase 5: SSO & Security (Weeks 10-11)
- [ ] SAML 2.0 integration
- [ ] OAuth 2.0 support
- [ ] LDAP authentication
- [ ] Security audit preparation (SOC 2)

### Phase 6: Enterprise Edition Release (Week 12)
- [ ] Package enterprise features
- [ ] Security audit checklist
- [ ] Enterprise deployment guide
- [ ] Pricing and licensing documentation

## Success Metrics

### Must Have
- [ ] Design versioning with diff/merge
- [ ] Complete audit trail (all operations)
- [ ] RBAC with 4+ roles
- [ ] Quota enforcement
- [ ] License management (JWT-based)
- [ ] SSO integration (SAML/OAuth)

### Should Have
- [ ] SOC 2 Type II audit preparation
- [ ] Floating license server
- [ ] Team collaboration features
- [ ] Usage analytics dashboard

### Nice to Have
- [ ] Blockchain-based audit log (immutable)
- [ ] AI-based anomaly detection
- [ ] Multi-tenancy support

## Dependencies

- All previous milestones (enterprise features integrate across the stack)
- M10 (I/O) - for efficient design storage
- M15 (Distributed) - for quota management
- M16 (Compliance) - for audit requirements

## References

- [SOC 2 Compliance](https://www.aicpa.org/interestareas/frc/assuranceadvisoryservices/aicpasoc2report.html)
- [SAML 2.0](http://docs.oasis-open.org/security/saml/Post2.0/sstc-saml-tech-overview-2.0.html)
- [OAuth 2.0](https://oauth.net/2/)
- [JWT (JSON Web Tokens)](https://jwt.io/)
