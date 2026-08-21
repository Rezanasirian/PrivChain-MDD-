# Incident: committed DAIC-WOZ text embeddings

- **Detected:** 2026-08-21
- **Status:** Containment in progress
- **Affected path:** `docs/runbook/text-embedding-cache/`
- **First known commit:** `f8c24d2`

## Summary

The repository history contains 282 NumPy embeddings derived from DAIC-WOZ
participant transcripts. Participant identifiers are present in filenames.
Sentence embeddings are data-derived artifacts and can be vulnerable to
inversion; they must be handled as sensitive data rather than documentation.

The affected commit is reachable from the local and remote
`phase-1/real-data-baseline` branch. No participant identifiers or file-level
inventory are reproduced in this note.

## Containment and remediation

1. Block data-like and model/checkpoint extensions everywhere in the repository
   with the pre-commit guard.
2. Create an encrypted, access-controlled backup outside the repository under
   the applicable DUA before deletion.
3. Remove the affected working-tree path and rewrite every affected Git ref with
   `git filter-repo`.
4. After explicit execution-time approval, temporarily adjust branch protection
   and force-push the rewritten affected refs.
5. Ask GitHub Support to purge cached unreachable objects and review forks, PR
   refs, Actions artifacts, and release assets.
6. Decide with the data custodian whether the DUA requires notification.

History rewriting does not close the prior exposure window. If the repository
was public or accessible to others, assume the objects may have been copied.

## Identifier policy

Committed artifacts must not contain raw participant IDs or unsalted hashes of
the small DAIC-WOZ identifier space. Reproducibility records use a whole-split
digest and, where linkage is required, an HMAC whose key remains server-side in
`.env`. The participant mapping never enters Git.
