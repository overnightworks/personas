# Runbook: show and erase what an agent keeps

Task: someone asks what an agent keeps of them, or that it be erased. Impact: the agent forgets every line,
table state, call and sentence of his history, and every kept line, of the agent or agents named.

Both commands load the runner's whole configuration, so `PERSONAS_CONFIG` names a valid one whose secret
files are all readable; a configuration it refuses ends them with exit 2. They use only its `[database]`
table and run under `write_seconds`.

## Check

`personas memory list [--agent <client id>]` prints, tab-separated, one row per visit (client id, app,
place, generation, lines kept, first and last read), a blank line, then one row per agent's history (client
id, history entries, answered turns, first and last entry). It never prints content.

## Fix

1. Wait until the agent has left the places concerned, or stop the runner: an erase removes what was kept
   up to that moment, and an agent still in a call keeps that call's window again at its next read.
2. `personas memory erase --agent <client id>`, or `--all` for every agent. Exactly one is given, spelled in
   full. It prints `<n> lines, <n> history entries and <n> answered turns erased`, 0 included.
3. Expect the pod serving him to stop him at his next write and serve him again from the erased history; a
   turn in between may still be told what was erased. Where reading his places stands is kept.
4. On `personas: memory: store_timeout` the erase deleted nothing, unless the bound ran out at the commit:
   run it again with a larger `write_seconds` and it erases what is left.

Rollback: none; an erase is irreversible.

## Escalate

A schema behind its head refuses both commands: run `personas migrate` with the release the runner runs.
Any other `personas: memory:` code is a database error: report it to the operator with the code.
