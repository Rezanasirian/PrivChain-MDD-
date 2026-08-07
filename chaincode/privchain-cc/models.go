package main

import "fmt"

// Object-type prefixes for composite ledger keys.
const (
	clientObjectType     = "client"
	budgetObjectType     = "budget"
	reputationObjectType = "reputation"
	subgraphObjectType   = "subgraph"
)

// modalities is the fixed capability-vector order used project-wide.
var modalities = [3]string{"audio", "video", "text"}

// CapabilityVector is the [audio, video, text] availability flags (each 0 or 1).
type CapabilityVector [3]int

// ClientRecord is a registered federated client and its declared capability.
// The owner fields bind the client to the identity that registered it, so only
// that identity (or the coordinator) may later write records about it.
type ClientRecord struct {
	ClientID         string           `json:"clientId"`
	Capability       CapabilityVector `json:"capability"`
	OwnerMSPID       string           `json:"ownerMspId"`
	OwnerFingerprint string           `json:"ownerFingerprint"`
}

// roundKeyWidth zero-pads the round component of a composite key. Composite
// keys iterate in lexicographic order, so a bare decimal round would order
// "10" before "2" and hand back an out-of-order audit history.
const roundKeyWidth = 10

// BudgetRecord is one append-only entry of per-modality privacy budget consumed
// by a client in a given round. Consumed epsilon is never overwritten
// (CLAUDE.md §7), so (clientId, modality, round) is unique.
type BudgetRecord struct {
	ClientID     string  `json:"clientId"`
	Modality     string  `json:"modality"`
	Round        int     `json:"round"`
	EpsilonSpent float64 `json:"epsilonSpent"`
}

// ReputationRecord is a client's latest per-modality reputation. Reputation is
// designed to evolve, so this record is updated in place (the round it was last
// set at is retained for audit).
type ReputationRecord struct {
	ClientID string  `json:"clientId"`
	Modality string  `json:"modality"`
	Score    float64 `json:"score"`
	Round    int     `json:"round"`
}

// SubgraphRecord is the set of clients aggregated together in a round. It is
// immutable once published (see ADR-0006).
type SubgraphRecord struct {
	Round     int      `json:"round"`
	ClientIDs []string `json:"clientIds"`
}

// roundKey renders a round number as a lexicographically sortable key part.
func roundKey(round int) string {
	return fmt.Sprintf("%0*d", roundKeyWidth, round)
}

// isKnownModality reports whether name is one of the three modalities.
func isKnownModality(name string) bool {
	for _, m := range modalities {
		if m == name {
			return true
		}
	}
	return false
}
