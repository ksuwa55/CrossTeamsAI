# False Negative Inspection

For each ground-truth pair the model missed: your original label, the single closest predicted pair from that meeting (even though it didn't match), and the judge's reasoning for why it fell short.

## meeting_01_signoff_blocker

**Your label:**
- cause: "the customer records module cannot be pushed"
- effect: "QA is holding off on writing test cases for that module"

**Closest predicted pair:**
- cause: "Delay in legal approval"
- effect: "Stalled progress on writing test cases"

**Verdict:** PARTIAL MATCH (captures part of the relationship)

**Judge reasoning:** The predicted pair captures the effect of stalled progress on writing test cases, which is related to the ground truth effect. However, it does not fully align with the cause, as it refers to a delay in legal approval rather than the inability to push the customer records module.

## meeting_02_ambiguous_requirement

**Your label:**
- cause: "the requirement is really unclear"
- effect: "there's been confusion about what 'guest checkout' actually means in the spec"

**Closest predicted pair:**
- cause: "Ambiguous requirement"
- effect: "Lost three days of work re-doing form validation logic"

**Verdict:** PARTIAL MATCH (captures part of the relationship)

**Judge reasoning:** The predicted pair captures the cause of an unclear requirement with 'Ambiguous requirement,' which is related but not identical to the ground truth. However, the effect 'Lost three days of work re-doing form validation logic' does not directly relate to the confusion about 'guest checkout' in the spec, making it a partial match.

## meeting_02_ambiguous_requirement

**Your label:**
- cause: "about three days of work have been lost re-doing the form validation logic"
- effect: "the team is going to miss the Friday deadline for the checkout release"

**Closest predicted pair:**
- cause: "Ambiguous requirement"
- effect: "Lost three days of work re-doing form validation logic"

**Verdict:** PARTIAL MATCH (captures part of the relationship)

**Judge reasoning:** The first predicted pair captures the effect of lost work but does not connect it to the specific consequence of missing the Friday deadline. It reflects a related issue of lost work but does not fully encompass the causal relationship described in the ground truth.

## meeting_03_missing_resource

**Your label:**
- cause: "QA is backed up"
- effect: "three modules are waiting for testing"

**Closest predicted pair:**
- cause: "Testing backlog due to lack of resources"
- effect: "Blocking every release"

**Verdict:** PARTIAL MATCH (captures part of the relationship)

**Judge reasoning:** The predicted pair captures the concept of a testing backlog, which relates to the cause of 'QA is backed up', but it does not specifically mention the effect of 'three modules are waiting for testing'. Instead, it refers to a broader effect of blocking every release, which is related but not the same.

## meeting_05_decision_stagnation

**Your label:**
- cause: "the team doesn't think they can decide without a proper load test"
- effect: "there is still no consensus and no decision has been made"

**Closest predicted pair:**
- cause: "Need for a proper load test"
- effect: "Delay in making a decision"

**Verdict:** PARTIAL MATCH (captures part of the relationship)

**Judge reasoning:** The predicted pair captures the cause of needing a proper load test, which aligns with the ground truth, but it does not fully capture the effect of no consensus and no decision being made. Instead, it suggests a delay in making a decision, which is related but not the same as the lack of consensus.

## meeting_06_timezone_communication

**Your label:**
- cause: "by the time Sam got to the review, the Osaka team had already logged off so he couldn't clarify a question he had"
- effect: "the Osaka team's branch cannot be merged because it's waiting on a code review from London"

**Closest predicted pair:**
- cause: "Code review pending from London team"
- effect: "Merge of branch delayed"

**Verdict:** PARTIAL MATCH (captures part of the relationship)

**Judge reasoning:** The predicted pair captures the effect of a delay in merging due to a pending code review, which is related to the ground truth. However, it does not include the specific cause of Sam not being able to clarify his question due to the Osaka team logging off, which is a critical part of the ground truth causal relationship.

## meeting_07_scope_change

**Your label:**
- cause: "the client added a new requirement for multi-currency support in the invoicing module"
- effect: "implementing multi-currency now touches the same code currently being integrated, creating conflicts with ongoing work"

**Closest predicted pair:**
- cause: "New requirement for currency feature"
- effect: "Delay in integration with accounting system"

**Verdict:** PARTIAL MATCH (captures part of the relationship)

**Judge reasoning:** The predicted pair captures the cause of a new requirement for a currency feature, which is related to the ground truth. However, the effect described in the predicted pair focuses on a delay in integration with the accounting system rather than the specific conflicts arising from the new requirement affecting the same code being integrated, which is a more nuanced aspect of the original effect.

## meeting_09_security_ambiguity

**Your label:**
- cause: "the compliance team is blocking the release over the undefined token expiry requirement"
- effect: "the team is blocked on compliance sign-off, while compliance is blocked on the team clarifying the requirement"

**Closest predicted pair:**
- cause: "Ambiguity regarding token expiration time"
- effect: "Compliance team blocking the release"

**Verdict:** PARTIAL MATCH (captures part of the relationship)

**Judge reasoning:** The first predicted pair captures the cause of the compliance team blocking the release due to ambiguity regarding the token expiration time, which is related to the ground truth. However, it does not fully capture the effect of the team being blocked on compliance sign-off, as it only mentions the compliance team blocking the release without addressing the reciprocal blocking situation.
