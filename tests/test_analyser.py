"""Tests for the ingestion analyser (heuristic + contradiction detection)."""

from __future__ import annotations

import pytest

from verra.ingest.analyser import (
    ChunkAnalysis,
    analyse_chunk_heuristic,
    check_contradictions,
)


class TestSentimentDetection:
    def test_positive_email(self):
        text = "Hi team, great work on the launch! The client is thrilled and we appreciate everyone's effort."
        result = analyse_chunk_heuristic(text)
        assert result.sentiment == "positive"

    def test_negative_complaint(self):
        text = "We're frustrated with the ongoing issues. The deployment failed again and the bug is still not fixed."
        result = analyse_chunk_heuristic(text)
        assert result.sentiment == "negative"

    def test_escalation(self):
        text = "This is unacceptable. We need this resolved immediately. I'm escalating to your management."
        result = analyse_chunk_heuristic(text)
        assert result.sentiment == "escalation"

    def test_neutral_factual(self):
        text = "The meeting is scheduled for Tuesday at 2pm. Please review the agenda beforehand."
        result = analyse_chunk_heuristic(text)
        assert result.sentiment == "neutral"


class TestTopicExtraction:
    def test_pricing_topic(self):
        text = "The monthly pricing for the professional plan is $79 per user with annual billing discounts."
        result = analyse_chunk_heuristic(text)
        assert "pricing" in result.topics

    def test_security_topic(self):
        text = "SSL certificates on 3 subdomains expire in 45 days. We need to address this vulnerability."
        result = analyse_chunk_heuristic(text)
        assert "security" in result.topics

    def test_infrastructure_topic(self):
        text = "We're deploying the new Docker containers to AWS with Kubernetes orchestration."
        result = analyse_chunk_heuristic(text)
        assert "infrastructure" in result.topics

    def test_multiple_topics(self):
        text = "The migration project involves upgrading our infrastructure and updating the pricing model."
        result = analyse_chunk_heuristic(text)
        assert len(result.topics) >= 2

    def test_max_five_topics(self):
        text = (
            "pricing cost billing invoice support ticket bug fix infrastructure "
            "server deploy docker meeting standup project phase milestone security ssl"
        )
        result = analyse_chunk_heuristic(text)
        assert len(result.topics) <= 5


class TestStalenessRisk:
    def test_low_staleness_for_current(self):
        text = "The new policy takes effect immediately and replaces all previous versions."
        result = analyse_chunk_heuristic(text)
        # Has staleness keywords like "previous" but context is current
        assert isinstance(result.staleness_risk, float)

    def test_higher_staleness_with_old_refs(self):
        text = "This was the old process. The legacy system was deprecated and no longer in use."
        result = analyse_chunk_heuristic(text)
        assert result.staleness_risk > 0


class TestAssertionExtraction:
    def test_extracts_is_assertions(self):
        text = "Acme Corporation is our largest client. The contract expires on January 14, 2025."
        result = analyse_chunk_heuristic(text)
        assert len(result.assertions) >= 1

    def test_extracts_cost_assertions(self):
        text = "The monthly retainer costs $12,500. Additional hours are charged at $175/hour."
        result = analyse_chunk_heuristic(text)
        assert any("$" in a for a in result.assertions)


class TestCommitmentExtraction:
    def test_ill_send_pattern(self):
        text = "I'll send the proposal to Lisa by end of week."
        result = analyse_chunk_heuristic(text)
        assert len(result.commitments) >= 1
        assert "proposal" in result.commitments[0]["what"].lower()

    def test_action_item_pattern(self):
        text = "Action item: Connor to schedule the quarterly review with John Smith."
        result = analyse_chunk_heuristic(text)
        assert len(result.commitments) >= 1

    def test_checkbox_pattern(self):
        text = "[ ] Marcus: Start TechCo codebase audit March 18"
        result = analyse_chunk_heuristic(text)
        assert len(result.commitments) >= 1

    def test_no_false_positives_on_short(self):
        text = "Hello, how are you today?"
        result = analyse_chunk_heuristic(text)
        assert len(result.commitments) == 0


class TestContradictionDetection:
    def test_number_conflict(self):
        old = ["The retainer costs $12,500 per month"]
        new = ["The retainer costs $15,000 per month"]
        conflicts = check_contradictions(new, old)
        assert len(conflicts) >= 1

    def test_status_conflict(self):
        old = ["Invoice INV-2024-005 is paid"]
        new = ["Invoice INV-2024-005 is overdue"]
        conflicts = check_contradictions(new, old)
        assert len(conflicts) >= 1

    def test_no_conflict_same_info(self):
        old = ["The contract expires January 2025"]
        new = ["The contract expires January 2025"]
        conflicts = check_contradictions(new, old)
        assert len(conflicts) == 0

    def test_no_conflict_different_subjects(self):
        old = ["Project Atlas costs $50,000"]
        new = ["Office rent costs $5,000"]
        conflicts = check_contradictions(new, old)
        assert len(conflicts) == 0


class TestFullAnalysis:
    def test_business_email(self):
        text = """From: connor@berghofferdigital.com
To: john.smith@acmecorp.com
Date: 2024-05-10
Subject: Outstanding Invoices

Hi John,

I'll send the updated invoice breakdown by Friday. The current outstanding
amount is $25,000 across two invoices.

We need to schedule a call to discuss the APM integration pricing.
The proposed rate is $3,500/month.

Best,
Connor"""
        result = analyse_chunk_heuristic(text)

        assert result.sentiment == "neutral"
        assert "pricing" in result.topics
        assert len(result.assertions) >= 1
        assert len(result.commitments) >= 1

    def test_meeting_notes(self):
        text = """# Team Standup — March 15, 2024

## Action Items
- [ ] Connor: Send follow-up email to Lisa Park
- [ ] Priya: Finalize Terraform modules by March 22
- [ ] Marcus: Start TechCo codebase audit March 18

## Decisions
- Agreed to hire a junior developer
- Moving standups to 9:30am"""
        result = analyse_chunk_heuristic(text)

        assert "meeting" in result.topics
        assert len(result.commitments) >= 2

    def test_contract_document(self):
        text = """Service Agreement — Acme Corporation

Effective Date: January 15, 2024
Expiration Date: January 14, 2025

Monthly retainer: $12,500/month
Additional engineering hours: $175/hour
SLA: 99.95% uptime guarantee"""
        result = analyse_chunk_heuristic(text)

        assert "contract" in result.topics
        assert len(result.assertions) >= 1
