"""Tests for expanded query intent classification."""

from verra.retrieval.router import QueryType, classify_query


class TestExpandedClassification:
    # Comparative
    def test_compare_clients(self):
        assert classify_query("How does Acme compare to TechCo on support volume?") == QueryType.COMPARATIVE

    def test_versus(self):
        assert classify_query("Acme vs TechCo revenue") == QueryType.COMPARATIVE

    def test_difference_between(self):
        assert classify_query("What's the difference between starter and pro plans?") == QueryType.COMPARATIVE

    # Temporal trends
    def test_improving(self):
        assert classify_query("Is our ticket resolution time improving?") == QueryType.TEMPORAL_TREND

    def test_over_time(self):
        assert classify_query("How has revenue changed over time?") == QueryType.TEMPORAL_TREND

    def test_trend(self):
        assert classify_query("What's the trend in support tickets?") == QueryType.TEMPORAL_TREND

    def test_declining(self):
        assert classify_query("Are client communications declining?") == QueryType.TEMPORAL_TREND

    # Hypothetical
    def test_what_if(self):
        assert classify_query("What if we lost the Acme contract?") == QueryType.HYPOTHETICAL

    def test_what_would_happen(self):
        assert classify_query("What would happen if we raised prices 20%?") == QueryType.HYPOTHETICAL

    def test_impact_of_losing(self):
        assert classify_query("What's the impact of losing client X?") == QueryType.HYPOTHETICAL

    # Meta queries
    def test_most_outdated(self):
        assert classify_query("What's the most outdated information in our system?") == QueryType.META

    def test_how_many_documents(self):
        assert classify_query("How many documents have been ingested?") == QueryType.META

    def test_what_sources(self):
        assert classify_query("What sources have been ingested?") == QueryType.META

    # Gap queries
    def test_what_missing(self):
        assert classify_query("What's missing from our documentation about Acme?") == QueryType.GAP

    def test_what_dont_we_have(self):
        assert classify_query("What don't we have documented about the pricing?") == QueryType.GAP

    def test_gaps(self):
        assert classify_query("What gaps exist in our client records?") == QueryType.GAP

    # Original types still work
    def test_semantic_still_works(self):
        assert classify_query("what is our refund policy?") == QueryType.SEMANTIC

    def test_hybrid_still_works(self):
        assert classify_query("what did Jake say about pricing?") == QueryType.HYBRID

    def test_metadata_still_works(self):
        qt = classify_query("emails from John last month")
        assert qt in (QueryType.METADATA, QueryType.HYBRID)


class TestKnowledgeGapDetection:
    def test_detects_missing_rationale(self):
        from verra.ingest.analyser import detect_knowledge_gaps
        text = "Contract renewed at 15% discount for Acme Corporation."
        gaps = detect_knowledge_gaps(text)
        assert len(gaps) >= 1
        assert gaps[0]["gap_type"] == "missing_rationale"

    def test_no_gap_with_rationale(self):
        from verra.ingest.analyser import detect_knowledge_gaps
        text = "Contract renewed at 15% discount because they committed to a 3-year term."
        gaps = detect_knowledge_gaps(text)
        assert len(gaps) == 0

    def test_detects_undocumented_decision(self):
        from verra.ingest.analyser import detect_knowledge_gaps
        text = "Decided to switch from AWS to GCP for all production workloads."
        gaps = detect_knowledge_gaps(text)
        assert len(gaps) >= 1


class TestEventExtraction:
    def test_deployment_event(self):
        from verra.ingest.analyser import extract_events
        events = extract_events("We deployed the new API gateway to production.")
        assert len(events) >= 1
        assert events[0]["event_type"] == "deployment"

    def test_hiring_event(self):
        from verra.ingest.analyser import extract_events
        events = extract_events("We hired a new senior developer starting Monday.")
        assert len(events) >= 1
        assert events[0]["event_type"] == "hiring"

    def test_incident_event(self):
        from verra.ingest.analyser import extract_events
        events = extract_events("There was a major outage affecting all production services.")
        assert len(events) >= 1
        assert events[0]["event_type"] == "incident"

    def test_no_events_in_normal_text(self):
        from verra.ingest.analyser import extract_events
        events = extract_events("The meeting is scheduled for Tuesday at 2pm.")
        assert len(events) == 0
