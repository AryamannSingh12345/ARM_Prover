"""`search.dag` — the Proof-DAG subsystem.

Introduced by the consolidation refactor (Phase 1). During the transition
`search.proof_dag` remains the public facade; new structure lands here
module by module. Phase 1 adds the first-class proof-obligation model
(`model`, `obligations`) alongside the existing `HaveNode`/`Sketch`
sketch contract — additive, nothing in `proof_dag` changes yet.
"""
