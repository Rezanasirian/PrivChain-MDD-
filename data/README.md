# `data/` — never committed

This directory holds datasets and is **git-ignored** except for this file and
`.gitkeep` (see [CLAUDE.md](../CLAUDE.md) §7).

- **Real DAIC-WOZ data must never be committed or pushed.** It is access-
  controlled (requires a Data Use Agreement) and is the longest-lead-time item
  in the project — apply early.
- **Mock data** for development/CI is generated on demand and written to
  `data/mock/` by:

  ```bash
  python scripts/generate_mock_data.py
  ```

  These files are also git-ignored; regenerate rather than commit them.
- Configure the real-data location via `DAIC_WOZ_ROOT` in your local `.env`
  (copy from `.env.example`).
