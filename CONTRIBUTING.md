# Contributing Guide

## Branch Strategy

```
main                        ← stable, reviewed code
├── phase1/summarization    ← Phase 1 development
│     └── feature/*         ← individual tasks
├── phase2/causal-modeling  ← Phase 2 development
│     └── feature/*         ← individual tasks
└── phase3/...              ← future phases
```

## Branch Naming Conventions

- `feature/<name>` — new functionality
- `fix/<name>` — bug fixes
- `experiment/<name>` — exploratory/research work

## Daily Operations

### 1. Starting New Work

```bash
# Switch to the relevant phase branch
git checkout phase1/summarization

# Create a feature branch from it
git checkout -b feature/improve-rouge-eval phase1/summarization
```

### 2. While Working

```bash
git add <files>
git commit -m "descriptive message"
```

### 3. When Feature Is Done

```bash
# Push feature branch
git push -u origin feature/improve-rouge-eval

# Create PR: feature → phase branch
gh pr create --base phase1/summarization
```

Review and merge the PR on GitHub.

### 4. Keep Phase Branch Updated Locally

```bash
git checkout phase1/summarization
git pull origin phase1/summarization
```

### 5. When a Phase Is Complete (Milestone)

```bash
# Create PR: phase → main
gh pr create --base main --head phase1/summarization
```

After merge, tag the milestone:

```bash
git checkout main
git pull
git tag v0.1-phase1
git push origin v0.1-phase1
```

### 6. Starting the Next Phase's Active Work

```bash
# Make sure main has the latest completed phase
git checkout phase2/causal-modeling
git merge main
git push origin phase2/causal-modeling
```

## Key Rules

- **Never commit directly to `main` or phase branches** — always go through feature branches and PRs.
- Phase branches contain the full codebase. Folder prefixes (`01_`, `02_`) indicate which phase each module belongs to.
- Feature branches should be short-lived. Merge them back into the phase branch promptly.

## PR Flow

```
feature/xxx  →  PR  →  phase1/summarization  →  PR  →  main
feature/yyy  →  PR  →  phase2/causal-modeling →  PR  →  main
```
