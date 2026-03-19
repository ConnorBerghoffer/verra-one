# TechCo Phase 1 Kickoff — April 22, 2024

**Attendees:** Connor Berghoffer, Marcus Wong (Lead Dev), Lisa Park (TechCo CTO), Dev Patel (TechCo Lead Engineer)

## Agenda

1. Introductions and team structure
2. Access and onboarding
3. Assessment approach
4. Communication plan
5. Timeline confirmation

## Notes

### Current State (Lisa)
- PHP monolith is 12 years old, ~350K lines of code
- Running on 3 bare-metal servers in a colo facility
- No containerization, manual deployments via SSH
- MySQL 5.7 database (single instance, no replication)
- Peak traffic: 50K concurrent users during sales events
- Major pain points: 4-hour deploy cycles, frequent downtime during deploys, can't scale horizontally

### Assessment Plan (Marcus)
- Week 1: Codebase audit — understand module boundaries, identify coupling
- Week 2: Infrastructure audit — current resource usage, bottlenecks
- Week 3: Domain modeling — map business domains to service boundaries
- Week 4: Architecture design + migration roadmap

### Access Needed
- GitHub repo access (Dev to set up by EOD)
- VPN credentials for accessing staging/production
- Database read-only access for schema analysis
- Monitoring dashboard access (they use Datadog)

### Communication Plan
- Weekly sync: Tuesdays 2pm AEST via Google Meet
- Slack channel: #techco-migration (both teams)
- Escalation: Lisa → Connor for blockers

## Action Items

- [ ] Dev Patel: Grant GitHub + VPN access by April 23
- [ ] Marcus: Begin codebase audit April 23
- [ ] Lisa: Share Datadog dashboard access
- [ ] Connor: Set up shared Linear project for Phase 1 tracking
