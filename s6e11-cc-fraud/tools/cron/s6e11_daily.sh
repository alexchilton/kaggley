#!/bin/bash
# Daily unattended S6E11 run, started by ~/Library/LaunchAgents/com.alexchilton.s6e11-daily.plist at 08:00.
# Every day until END: read the competition's public code and discussion, write docs/cron/<date>.md, and if anything is
# worth adding, build, push and submit a new notebook version the same day.
# Stops for good once the notebook has silver: counted.json (Expert votes, read by Alex) says tier silver or counted >= 20.
END=2026-11-30           # competition closes 2026-11-30
SLUG=credit-card-fraud-lb-0-9515-shallow-tree-stack
COUNTED=/Users/alexchilton/PycharmProjects/kaggle-board/counted.json
REPO=/Users/alexchilton/PycharmProjects/playground-s6e11
MAX_SECONDS=10800        # 3 hours per run, hard stop (an update run waits for the Kaggle commit)

TODAY=$(date +%F)
cd "$REPO" || exit 1
mkdir -p logs/cron docs/cron scratch/cron
LOG="logs/cron/$TODAY.log"
if [[ "$TODAY" > "$END" ]]; then
    echo "[$(date +%T)] after $END, not running" >> logs/cron/skipped.log
    exit 0
fi
if python3 -c "import json,sys; d=json.load(open('$COUNTED')).get('$SLUG',{}); sys.exit(0 if d.get('tier')=='silver' or d.get('counted',0)>=20 else 1)" 2>/dev/null; then
    echo "[$(date +%T)] silver reached per $COUNTED, not running" >> logs/cron/skipped.log
    exit 0
fi
LOCK="$REPO/scratch/cron/run.lock"
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
    echo "[$(date +%T)] previous run PID $(cat "$LOCK") still alive, skipping" >> "$LOG"
    exit 0
fi
echo $$ > "$LOCK"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$HOME/Library/Python/3.12/bin:$PATH"
echo "[$(date +%T)] start, claude $(claude --version 2>&1 | head -1)" >> "$LOG"
perl -e 'alarm shift; exec @ARGV' "$MAX_SECONDS" \
    claude -p "$(cat tools/cron/daily_prompt.md)

Today is $TODAY. Your report path is docs/cron/$TODAY.md." \
    --allowedTools "Bash Read Write Edit Glob Grep WebFetch" \
    --permission-mode acceptEdits \
    --output-format text >> "$LOG" 2>&1
echo "[$(date +%T)] exit $?" >> "$LOG"
git add -A docs/cron && ! git diff --cached --quiet && git commit -q -m "cron $TODAY: report" -- docs/cron >> "$LOG" 2>&1
# copy the tracked files to the public kaggley repo (github.com/alexchilton/kaggley/s6e11-cc-fraud), where the playgrounds live
KY=/Users/alexchilton/PycharmProjects/kaggley
git ls-files | grep -v '^.gitignore$' | rsync -a --files-from=- . "$KY/s6e11-cc-fraud/" >> "$LOG" 2>&1
(cd "$KY" && git add s6e11-cc-fraud && ! git diff --cached --quiet && git commit -q -m "s6e11-cc-fraud: sync $TODAY" && git push -q) >> "$LOG" 2>&1
rm -f "$LOCK"
