---
doc_id: incident-response-runbook
title: Security Incident Response Runbook
allowed_roles: [security]
classification: restricted
sensitive: true
owner: Security Operations
effective: 2026-02-15
---
# Severity Levels
Severity 1 means confirmed customer data exposure or a production outage and is paged to the on-call engineer within 15 minutes. Severity 2 means a contained compromise of an internal system and must be triaged within 4 hours. Severity 3 covers policy violations without data exposure and is handled the next business day.

# Containment and Evidence
Isolate affected hosts from the network but never power them off before a forensic image is captured. Rotate all credentials that touched the affected systems. The incident commander records every action in the incident log with a timestamp.

# Notification
The CISO must be notified of every Severity 1 incident within 1 hour. Regulators must be notified of a confirmed personal data breach within 72 hours of discovery. The break-glass account procedure is stored in the vault under the path secret/ir/breakglass and requires two approvers.
