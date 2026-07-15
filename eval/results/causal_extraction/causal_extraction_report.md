# Causal Event Extraction Evaluation

Extraction model: `gpt-3.5-turbo` (unchanged pipeline). Matching: LLM judge (`gpt-4o-mini`), one-to-one maximum bipartite matching per meeting.

## Per-meeting results

| Meeting | Predicted | Gold | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| meeting_01_signoff_blocker | 7 | 3 | 2 | 5 | 1 | 28.6% | 66.7% | 40.0% |
| meeting_02_ambiguous_requirement | 3 | 3 | 1 | 2 | 2 | 33.3% | 33.3% | 33.3% |
| meeting_03_missing_resource | 2 | 1 | 0 | 2 | 1 | 0.0% | 0.0% | 0.0% |
| meeting_04_api_dependency | 8 | 1 | 1 | 7 | 0 | 12.5% | 100.0% | 22.2% |
| meeting_05_decision_stagnation | 5 | 1 | 0 | 5 | 1 | 0.0% | 0.0% | 0.0% |
| meeting_06_timezone_communication | 6 | 2 | 1 | 5 | 1 | 16.7% | 50.0% | 25.0% |
| meeting_07_scope_change | 4 | 2 | 1 | 3 | 1 | 25.0% | 50.0% | 33.3% |
| meeting_08_qa_delay_deadline | 6 | 2 | 2 | 4 | 0 | 33.3% | 100.0% | 50.0% |
| meeting_09_security_ambiguity | 2 | 2 | 1 | 1 | 1 | 50.0% | 50.0% | 50.0% |
| meeting_10_resource_reprioritization | 7 | 2 | 2 | 5 | 0 | 28.6% | 100.0% | 44.4% |
| **Aggregate (pooled)** | 50 | 19 | 11 | 39 | 8 | 22.0% | 57.9% | 31.9% |

*Aggregate is pooled (micro-averaged): TP/FP/FN are summed across all meetings before computing precision/recall/F1.*

## False positives (predicted, no matching ground truth)

- **meeting_01_signoff_blocker**: cause="Data-sharing agreement not signed off" -> effect="Blocking development of customer records module" (t=00:00:45)
- **meeting_01_signoff_blocker**: cause="Pending approval for cross-border data transfer" -> effect="Blocking push of customer records module" (t=00:00:45)
- **meeting_01_signoff_blocker**: cause="Delay in legal approval" -> effect="Stalled progress on writing test cases" (t=00:01:10)
- **meeting_01_signoff_blocker**: cause="Pending legal approval" -> effect="Unsustainable sprint progress" (t=00:02:15)
- **meeting_01_signoff_blocker**: cause="Delay in approval from legal team" -> effect="Stalled integration timeline with Berlin team" (t=00:02:50)
- **meeting_02_ambiguous_requirement**: cause="Scope shifting" -> effect="Likely deadline slip" (t=00:02:40)
- **meeting_02_ambiguous_requirement**: cause="Ambiguous spec update" -> effect="Redoing work next week" (t=00:03:00)
- **meeting_03_missing_resource**: cause="Testing backlog due to lack of resources" -> effect="Blocking every release" (t=00:01:35)
- **meeting_03_missing_resource**: cause="Testing backlog blocking releases" -> effect="Delay in releasing new features" (t=00:01:35)
- **meeting_04_api_dependency**: cause="Delay in receiving sandbox API credentials from shipping vendor" -> effect="Delay in integration testing phase" (t=00:01:20)
- **meeting_04_api_dependency**: cause="No response from support channel" -> effect="Integration testing phase delayed" (t=00:01:20)
- **meeting_04_api_dependency**: cause="Lack of live API access" -> effect="Integration testing phase delay" (t=00:01:20)
- **meeting_04_api_dependency**: cause="Inability to fully validate without real API credentials" -> effect="End-to-end integration delay" (t=00:02:00)
- **meeting_04_api_dependency**: cause="Integration delay extending another week" -> effect="Pushing into next sprint's launch window" (t=00:02:20)
- **meeting_04_api_dependency**: cause="Delay in receiving real credentials" -> effect="Integration delay" (t=00:02:00)
- **meeting_04_api_dependency**: cause="Integration delay pushing into next sprint's launch window" -> effect="Missed launch window" (t=00:02:20)
- **meeting_05_decision_stagnation**: cause="Disagreement on database choice" -> effect="Delay in finalizing database selection" (t=00:00:20)
- **meeting_05_decision_stagnation**: cause="Unresolved concerns about Postgres scaling" -> effect="Lack of consensus on database choice" (t=00:01:00)
- **meeting_05_decision_stagnation**: cause="Need for a proper load test" -> effect="Delay in making a decision" (t=00:01:20)
- **meeting_05_decision_stagnation**: cause="indecisiveness on architecture" -> effect="blocking the analytics roadmap" (t=00:01:40)
- **meeting_05_decision_stagnation**: cause="indecisiveness on architecture" -> effect="missing the MVP deadline" (t=00:02:00)
- **meeting_06_timezone_communication**: cause="Code review pending from London team" -> effect="Merge of branch delayed" (t=00:00:20)
- **meeting_06_timezone_communication**: cause="Communication delay due to time zone difference" -> effect="Blocking progress on tasks" (t=00:01:20)
- **meeting_06_timezone_communication**: cause="Timezone gap for reviews" -> effect="Delays in reviews" (t=00:02:20)
- **meeting_06_timezone_communication**: cause="Setting up a dedicated async channel for review questions" -> effect="Addressing delays in reviews" (t=00:02:40)
- **meeting_06_timezone_communication**: cause="Timezone gap for review questions" -> effect="Day of delay for every review" (t=00:02:20)
- **meeting_07_scope_change**: cause="New requirement for currency feature" -> effect="Delay in integration with accounting system" (t=00:01:20)
- **meeting_07_scope_change**: cause="Pushing the currency feature to the next sprint" -> effect="Accounting integration slipping by at least a week" (t=00:02:20)
- **meeting_07_scope_change**: cause="Client urgency" -> effect="Accounting integration delay" (t=00:02:20)
- **meeting_08_qa_delay_deadline**: cause="QA backlog growing" -> effect="Missing the deadline" (t=00:01:40)
- **meeting_08_qa_delay_deadline**: cause="Freezing new features" -> effect="Possibility of hitting the deadline" (t=00:01:40)
- **meeting_08_qa_delay_deadline**: cause="Testing backlog" -> effect="Risk of missing deadline" (t=00:01:40)
- **meeting_08_qa_delay_deadline**: cause="Freezing new feature work" -> effect="Possibility to clear backlog by Wednesday" (t=00:02:40)
- **meeting_09_security_ambiguity**: cause="Client's security lead not responding to proposed token expiration time" -> effect="Delay in resolving the unclear requirement" (t=00:01:20)
- **meeting_10_resource_reprioritization**: cause="Carlos being out sick" -> effect="Dependency on notifications service" (t=00:00:20)
- **meeting_10_resource_reprioritization**: cause="Carlos being out sick" -> effect="Inability to finish the notifications feature in time for the demo" (t=00:00:40)
- **meeting_10_resource_reprioritization**: cause="Lack of documentation" -> effect="Inability to finish the notifications feature in time for the demo on Friday" (t=00:01:20)
- **meeting_10_resource_reprioritization**: cause="Lack of documentation" -> effect="Delay in feature completion" (t=00:01:40)
- **meeting_10_resource_reprioritization**: cause="Lack of documentation" -> effect="Repeating the same issue" (t=00:03:00)

## False negatives (ground truth missed by the model)

- **meeting_01_signoff_blocker**: cause="the customer records module cannot be pushed" -> effect="QA is holding off on writing test cases for that module" (t=00:01:10)
- **meeting_02_ambiguous_requirement**: cause="the requirement is really unclear" -> effect="there's been confusion about what 'guest checkout' actually means in the spec" (t=00:00:00)
- **meeting_02_ambiguous_requirement**: cause="about three days of work have been lost re-doing the form validation logic" -> effect="the team is going to miss the Friday deadline for the checkout release" (t=00:02:20)
- **meeting_03_missing_resource**: cause="QA is backed up" -> effect="three modules are waiting for testing" (t=00:00:35)
- **meeting_05_decision_stagnation**: cause="the team doesn't think they can decide without a proper load test" -> effect="there is still no consensus and no decision has been made" (t=00:01:00)
- **meeting_06_timezone_communication**: cause="by the time Sam got to the review, the Osaka team had already logged off so he couldn't clarify a question he had" -> effect="the Osaka team's branch cannot be merged because it's waiting on a code review from London" (t=00:00:20)
- **meeting_07_scope_change**: cause="the client added a new requirement for multi-currency support in the invoicing module" -> effect="implementing multi-currency now touches the same code currently being integrated, creating conflicts with ongoing work" (t=00:00:40)
- **meeting_09_security_ambiguity**: cause="the compliance team is blocking the release over the undefined token expiry requirement" -> effect="the team is blocked on compliance sign-off, while compliance is blocked on the team clarifying the requirement" (t=00:01:00)
