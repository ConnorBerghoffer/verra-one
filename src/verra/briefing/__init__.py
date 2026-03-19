"""Briefing system for Verra.

Analyses ingested data and surfaces actionable insights:
  - stale leads (sent emails with no reply after N days)
  - expiring contracts (documents with approaching valid_until dates)
  - forgotten commitments (open commitments that are overdue or approaching)
  - recurring patterns (topics appearing repeatedly in recent chat sessions)
  - new data summary (documents and emails ingested since last briefing)
"""


from verra.briefing.detector import BriefingDetector, BriefingItem

__all__ = ["BriefingDetector", "BriefingItem"]
