# Knowledge Graph Extraction Evaluation

Extraction model: `gpt-3.5-turbo` (unchanged pipeline). Triple matching: LLM judge (`gpt-4o-mini`), one-to-one maximum bipartite matching per meeting.

## Entity-linking (Top-N precision/recall)

| Meeting | Pred Entities | Gold Entities | Matched | Precision | Recall |
|---|---|---|---|---|---|
| meeting_01_signoff_blocker | 16 | 5 | 0 | 0.0% | 0.0% |
| meeting_02_ambiguous_requirement | 15 | 6 | 2 | 13.3% | 33.3% |
| meeting_03_missing_resource | 4 | 6 | 1 | 25.0% | 16.7% |
| meeting_04_api_dependency | 10 | 6 | 3 | 30.0% | 50.0% |
| meeting_05_decision_stagnation | 16 | 6 | 3 | 18.8% | 50.0% |
| meeting_06_timezone_communication | 16 | 7 | 3 | 18.8% | 42.9% |
| meeting_07_scope_change | 9 | 6 | 1 | 11.1% | 16.7% |
| meeting_08_qa_delay_deadline | 6 | 6 | 1 | 16.7% | 16.7% |
| meeting_09_security_ambiguity | 5 | 6 | 1 | 20.0% | 16.7% |
| meeting_10_resource_reprioritization | 14 | 7 | 2 | 14.3% | 28.6% |
| **Aggregate (pooled)** | 111 | 61 | 17 | 15.3% | 27.9% |

## Relation / fact extraction (Precision / Recall / F1)

| Meeting | Predicted | Gold | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| meeting_01_signoff_blocker | 9 | 4 | 1 | 8 | 3 | 11.1% | 25.0% | 15.4% |
| meeting_02_ambiguous_requirement | 8 | 4 | 1 | 7 | 3 | 12.5% | 25.0% | 16.7% |
| meeting_03_missing_resource | 2 | 4 | 1 | 1 | 3 | 50.0% | 25.0% | 33.3% |
| meeting_04_api_dependency | 7 | 4 | 2 | 5 | 2 | 28.6% | 50.0% | 36.4% |
| meeting_05_decision_stagnation | 10 | 4 | 2 | 8 | 2 | 20.0% | 50.0% | 28.6% |
| meeting_06_timezone_communication | 12 | 4 | 1 | 11 | 3 | 8.3% | 25.0% | 12.5% |
| meeting_07_scope_change | 7 | 4 | 1 | 6 | 3 | 14.3% | 25.0% | 18.2% |
| meeting_08_qa_delay_deadline | 3 | 4 | 0 | 3 | 4 | 0.0% | 0.0% | 0.0% |
| meeting_09_security_ambiguity | 3 | 4 | 1 | 2 | 3 | 33.3% | 25.0% | 28.6% |
| meeting_10_resource_reprioritization | 10 | 4 | 2 | 8 | 2 | 20.0% | 50.0% | 28.6% |
| **Aggregate (pooled)** | 71 | 40 | 12 | 59 | 28 | 16.9% | 30.0% | 21.6% |

*Aggregate is pooled (micro-averaged): TP/FP/FN (or matched/pred/gold for entity linking) are summed across all meetings before computing precision/recall/F1.*

## Graph coherence (unsupervised, no gold labels)

Built from all predicted triples pooled across meetings: 110 nodes, 71 edges, density=0.0059, connected components=40, communities=40, modularity=0.9575510204081618, avg. inter-community conductance=0.0.

## False positives (predicted, no matching ground truth)

- **meeting_01_signoff_blocker**: writing test cases --[depends_on]--> sign-off on the cross-border data transfer
- **meeting_01_signoff_blocker**: writing test cases --[depends_on]--> legal response
- **meeting_01_signoff_blocker**: writing test cases for that module --[depends_on]--> legal response
- **meeting_01_signoff_blocker**: approval for pending task --[blocks]--> progress in the sprint
- **meeting_01_signoff_blocker**: approval --[blocks]--> sprint progress
- **meeting_01_signoff_blocker**: integration timeline with the Berlin team --[blocks]--> progress
- **meeting_01_signoff_blocker**: integration timeline with the Berlin team --[blocks]--> progress
- **meeting_01_signoff_blocker**: team --[discusses]--> reporting dashboard tickets
- **meeting_02_ambiguous_requirement**: email assumption --[causes]--> scope change
- **meeting_02_ambiguous_requirement**: requirement reinterpretation --[causes]--> estimates delay
- **meeting_02_ambiguous_requirement**: ambiguity --[causes]--> work delay
- **meeting_02_ambiguous_requirement**: team --[raises_issue]--> client confirmation
- **meeting_02_ambiguous_requirement**: Tomas - Engineer --[raises_issue]--> missing Friday deadline for the checkout release
- **meeting_02_ambiguous_requirement**: Sofia - PM --[owns]--> telling the client about the likely deadline slip
- **meeting_02_ambiguous_requirement**: Mei - Engineer --[raises_issue]--> ambiguity in the spec update
- **meeting_03_missing_resource**: Diego - Engineer --[blocks]--> every release
- **meeting_04_api_dependency**: sandbox API credentials --[blocks]--> integration testing phase
- **meeting_04_api_dependency**: shipping vendor --[blocks]--> Yuki - Engineer
- **meeting_04_api_dependency**: Yuki - Engineer --[blocks]--> live API access
- **meeting_04_api_dependency**: Marco - PM --[resolves]--> being fully stuck
- **meeting_04_api_dependency**: API access --[blocks]--> simulate an order
- **meeting_05_decision_stagnation**: Elena - Host --[makes_decision]--> finalize which database to use for analytics service
- **meeting_05_decision_stagnation**: Raj - Engineer --[discusses]--> Postgres as safer choice
- **meeting_05_decision_stagnation**: Tokyo team --[discusses]--> managed NoSQL option
- **meeting_05_decision_stagnation**: Raj - Engineer --[discusses]--> Postgres vs. NoSQL
- **meeting_05_decision_stagnation**: Kenji - Engineer --[raises_issue]--> Postgres scaling for write volume
- **meeting_05_decision_stagnation**: Elena - Host --[makes_decision]--> urgency for decision
- **meeting_05_decision_stagnation**: Raj - Engineer --[blocks]--> decision without load test
- **meeting_05_decision_stagnation**: decision --[depends_on]--> load test
- **meeting_06_timezone_communication**: Osaka team --[blocks]--> merging branch
- **meeting_06_timezone_communication**: London team --[assigned_to]--> review Osaka team's code
- **meeting_06_timezone_communication**: Sam - Engineer --[blocks]--> Naoki - Engineer
- **meeting_06_timezone_communication**: Naoki - Engineer --[blocks]--> team
- **meeting_06_timezone_communication**: Fatima - PM --[discusses]--> setting up an overlap window or an async comment thread
- **meeting_06_timezone_communication**: Sam - Engineer --[raises_issue]--> question clarification delay
- **meeting_06_timezone_communication**: Naoki - Engineer --[blocks]--> waiting on Sam's reply
- **meeting_06_timezone_communication**: Fatima - PM --[makes_decision]--> async comment thread
- **meeting_06_timezone_communication**: Sam - Engineer --[owns]--> async comment thread suggestion
- **meeting_06_timezone_communication**: Naoki - Engineer --[raises_issue]--> timezone gap causing delay in reviews
- **meeting_06_timezone_communication**: Fatima - PM --[makes_decision]--> setting up a dedicated async channel for review questions
- **meeting_07_scope_change**: integration with the accounting system --[depends_on]--> new requirement
- **meeting_07_scope_change**: currency feature --[blocks]--> accounting integration
- **meeting_07_scope_change**: currency feature --[makes_decision]--> push the currency feature to the next sprint
- **meeting_07_scope_change**: currency feature --[resolves]--> accounting integration slipping by at least a week
- **meeting_07_scope_change**: currency feature --[depends_on]--> accounting integration
- **meeting_07_scope_change**: accounting integration --[blocks]--> sprint
- **meeting_08_qa_delay_deadline**: testing --[depends_on]--> fifteen test cases
- **meeting_08_qa_delay_deadline**: Omar - QA --[makes_decision]--> freeze new merges
- **meeting_08_qa_delay_deadline**: Ines - Engineer --[discusses]--> freeze feature work
- **meeting_09_security_ambiguity**: tokens expiration --[resolves]--> ambiguity
- **meeting_09_security_ambiguity**: tokens expiration --[resolves]--> client's security lead not responding
- **meeting_10_resource_reprioritization**: Carlos --[related_to]--> notifications service
- **meeting_10_resource_reprioritization**: Carlos --[blocks]--> notifications feature
- **meeting_10_resource_reprioritization**: notifications feature --[depends_on]--> resource gap
- **meeting_10_resource_reprioritization**: documentation --[resolves]--> pick up where he left off
- **meeting_10_resource_reprioritization**: Carlos --[related_to]--> issue of lack of documentation
- **meeting_10_resource_reprioritization**: dashboard improvements --[depends_on]--> demo
- **meeting_10_resource_reprioritization**: Carlos --[assigned_to]--> document his work
- **meeting_10_resource_reprioritization**: Carlos --[related_to]--> leave

## False negatives (ground truth missed by the model)

- **meeting_01_signoff_blocker**: Aiko - PM --[raises_issue]--> legal sign-off pending on data-sharing agreement
- **meeting_01_signoff_blocker**: Liam - Engineer --[owns]--> customer records module
- **meeting_01_signoff_blocker**: Aiko - PM --[makes_decision]--> escalate to legal via email and director
- **meeting_02_ambiguous_requirement**: Sofia - PM --[raises_issue]--> guest checkout requirement is ambiguous
- **meeting_02_ambiguous_requirement**: guest checkout requirement is ambiguous --[causes]--> checkout release may miss Friday deadline
- **meeting_02_ambiguous_requirement**: Mei - Engineer --[owns]--> checkout form validation logic
- **meeting_03_missing_resource**: Priya - QA --[raises_issue]--> QA is understaffed with one person covering all testing
- **meeting_03_missing_resource**: QA is understaffed with one person covering all testing --[blocks]--> payments module release
- **meeting_03_missing_resource**: Priya - QA --[owns]--> payments module testing
- **meeting_04_api_dependency**: Marco - PM --[makes_decision]--> mock the API response until real credentials arrive
- **meeting_04_api_dependency**: Marco - PM --[assigned_to]--> escalate credentials issue with shipping vendor account manager
- **meeting_05_decision_stagnation**: Kenji - Engineer --[raises_issue]--> no consensus on analytics database choice
- **meeting_05_decision_stagnation**: Raj - Engineer --[makes_decision]--> timebox load test to three days and commit to final decision
- **meeting_06_timezone_communication**: Naoki - Engineer --[raises_issue]--> branch blocked waiting on London code review
- **meeting_06_timezone_communication**: Sam - Engineer --[assigned_to]--> code review for Osaka branch
- **meeting_06_timezone_communication**: twelve hour timezone gap --[causes]--> branch blocked waiting on London code review
- **meeting_07_scope_change**: Olivia - PM --[raises_issue]--> client added multi-currency support as new scope
- **meeting_07_scope_change**: client added multi-currency support as new scope --[causes]--> accounting integration delayed
- **meeting_07_scope_change**: Ben - Engineer --[owns]--> invoicing accounting system integration
- **meeting_08_qa_delay_deadline**: Omar - QA --[raises_issue]--> QA backlog of fifteen untested cases
- **meeting_08_qa_delay_deadline**: QA backlog of fifteen untested cases --[blocks]--> Thursday release
- **meeting_08_qa_delay_deadline**: Grace - Host --[makes_decision]--> freeze new merges until QA backlog clears
- **meeting_08_qa_delay_deadline**: Omar - QA --[owns]--> clearing the QA testing backlog
- **meeting_09_security_ambiguity**: Wei - Engineer --[raises_issue]--> token expiry requirement was never clearly defined
- **meeting_09_security_ambiguity**: Anders - PM --[makes_decision]--> proceed with 15 minutes plus refresh tokens unless objections by Friday
- **meeting_09_security_ambiguity**: compliance sign-off --[depends_on]--> token expiry requirement was never clearly defined
- **meeting_10_resource_reprioritization**: Felix - Engineer --[raises_issue]--> Carlos is out sick and is the only one who understands the notifications service
- **meeting_10_resource_reprioritization**: Carlos is out sick and is the only one who understands the notifications service --[causes]--> notifications feature will miss Friday deadline
