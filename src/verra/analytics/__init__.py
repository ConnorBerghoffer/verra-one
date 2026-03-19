"""Verra analytics package.

Provides batch analytics that aggregate data across the full knowledge base
and store pre-computed results in SQLite for fast retrieval by the chat engine
and dashboard.
"""


from verra.analytics.batch import BatchAnalytics

__all__ = ["BatchAnalytics"]
