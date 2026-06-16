# CLAUDE.md — 12-Rule Template

These rules apply to every task in this project. When in conflict, prioritize safety, correctness, and simplicity.

### Rule 1 — Think Before Coding

State assumptions explicitly. Ask questions if a requirement is ambiguous. Push back when a simpler or better approach exists, and do not proceed when confused.

### Rule 2 — Simplicity First

Write the minimum code that solves the problem. Nothing speculative. No over-engineering or extra abstractions for single-use code. Code should be boring and highly readable.

### Rule 3 — Surgical Changes

Touch only what the task requires. Do not improve adjacent code, and never refactor what isn't broken. Every changed line should be directly traceable to the user's request.

### Rule 4 — Goal-Driven Execution

Define concrete success criteria before writing a single line of code. Loop autonomously until those specific success criteria are verified. Rely on robust testing to prove the feature works.

### Rule 5 — Use Code to Answer

Use the model only for judgment calls: classification, drafting, summarization, or extraction. Do NOT use the model for deterministic transforms; if code (or standard CLI tools) can reliably calculate, search, or answer, let code answer.

### Rule 6 — Token Budget Control

Per-task context limits are strict. If a complex task causes context bloat, summarize accomplishments and start a fresh thread to avoid context window degradation. Surface the breach—do not silently overrun.

### Rule 7 — Surface Conflicts

Don't average contradictory patterns. If there are two competing patterns, pick the more recent/tested one, explain why, and flag the outdated one for cleanup.

### Rule 8 — Read Before You Write

Thoroughly read relevant existing codebase components (e.g., immediate callers, shared utilities, exports) before generating new code. Do not build on top of deprecated stubs or disconnected entry points.

### Rule 9 — Multi-Step Checkpointing

For multi-step or complex refactors, create checkpoints after each significant stage. Summarize accomplishments, verify states, and pause if re-verification fails before proceeding.

### Rule 10 — Test Meaningfully

Ensure all written tests are meaningful and directly exercise the intended edge cases. Do not write dummy tests that merely pass to give an illusion of coverage.

### Rule 11 — Document As You Learn

When you encounter and resolve a complex debugging issue, summarize the root cause and solution. If needed, use sub-agents for heavy lifting, but keep your final learnings dated in your project memory.

### Rule 12 — Graceful Offloading

Some tasks belong to specialized agents. If a task scales beyond the reasonable scope of a single thread (e.g., a massive security overhaul or a multi-service migration), split it up and spin off targeted sub-agents.

### Rule 13 - Git Worktree Discipline

Never work directly on the main branch (only when specified, you can). Always create a new Git worktree or an isolated branch for any implementation or change, ensuring that all development occurs outside of main. Each task or feature must be developed in its own dedicated worktree or branch, keeping changes well organized and properly isolated. Commit changes incrementally and logically per task or subtask, maintaining a clean and structured commit history. The main branch must remain untouched until all changes are fully reviewed and ready for merge.

The dedicated worktree directory should be located at the parent folder of the repo and named: "working_trees/"

Create the required worktrees and perform commits using clear, objective titles along with concise descriptions that summarize what was implemented at each step. Avoid overly detailed explanations in commit messages, keeping them direct and informative. At the end of the process, update/create the PR for the `release` branch with the changes made.
