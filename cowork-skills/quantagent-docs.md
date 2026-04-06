# quantagent-docs — Documentation Manager Skill

## Role
You are the Documentation Manager for QuantAgent v2. Your job is to ensure PROJECT_CONTEXT.md stays synchronized with the actual codebase.

## When Triggered
- Scheduled: Daily at 23:30 (after development day ends)
- On-demand: When asked to "sync docs" or "update documentation"

## Workspace
Project directory: ~/quantagent-v2/

## Steps

### 1. Scan actual file structure
```bash
cd ~/quantagent-v2
find . -name "*.py" -not -path "./.venv/*" -not -path "./__pycache__/*" | sort
```

### 2. Compare to PROJECT_CONTEXT.md Section 3 (Project Structure)
Read `~/quantagent-v2/PROJECT_CONTEXT.md` and compare the documented file tree against actual files.
- Files that exist but aren't documented → add them
- Files that are documented but don't exist → flag as stale

### 3. Scan module implementation status
For each module listed in Section 2 (Module Inventory), check if the file:
- Exists and has more than just a docstring (actually implemented)
- Has corresponding test files in tests/

Compare against the STATUS column. Flag discrepancies:
- Module marked NOT BUILT but file has real implementation → should be IMPLEMENTED
- Module marked IMPLEMENTED but file is empty/stub → should be NOT BUILT

### 4. Count and verify tests
```bash
cd ~/quantagent-v2
python -m pytest tests/ --collect-only -q 2>/dev/null | tail -1
```

Check if the test count in the changelog matches the actual count.

### 5. Check CHANGELOG.md
Read `~/quantagent-v2/CHANGELOG.md` (if it exists).
Check if the last entry matches recent git commits:
```bash
cd ~/quantagent-v2
git log --oneline -5 2>/dev/null
```

### 6. Generate diff report
If any discrepancies found, create a summary:

```
Documentation Sync Report — [DATE]

DISCREPANCIES FOUND:

Section 2 (Module Inventory):
- [module_name]: file has implementation but status says NOT BUILT → update to IMPLEMENTED
- [module_name]: file is empty stub but status says IMPLEMENTED → update to NOT BUILT

Section 3 (Project Structure):
- NEW FILES not documented: [list]
- STALE entries (file doesn't exist): [list]

Section 14 (Changelog):
- Last changelog entry: [date]
- Commits since then: [count]
- Changelog may be stale.

Test count: documented=[N], actual=[N]
```

### 7. Fix discrepancies
If discrepancies found, update PROJECT_CONTEXT.md:
- Update Section 2 module statuses
- Update Section 3 file tree
- Update Section 14 changelog if git commits exist that aren't logged
- Update test count references

### 8. Post to Slack
If changes were made, post to #sprint-updates:
```
:memo: Docs synced — [DATE]

[N] discrepancies fixed in PROJECT_CONTEXT.md:
- [brief summary of changes]
```

If no changes needed:
```
:white_check_mark: Docs in sync — [DATE]
No discrepancies found.
```

## Important Rules
- Only modify PROJECT_CONTEXT.md. Never touch code files.
- Never change CLAUDE.md or ARCHITECTURE.md — those are architect-managed.
- Always create a git commit after modifying docs: `git add PROJECT_CONTEXT.md && git commit -m "docs: sync PROJECT_CONTEXT.md [auto]"`
- Be conservative — if unsure whether a module is "implemented", check for actual logic (not just imports and class stubs).
