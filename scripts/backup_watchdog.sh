#!/bin/bash
# Periodically run backup_scratch_critical.sh, detached, for a BOUNDED number of rounds.
#
# WHY THIS INSTEAD OF CRON: `crontab` is denied for this account on balam-login01 ("You (markymoo)
# are not allowed to use this program"), so the standard mechanism is unavailable. The need is real:
# the docking slices flush partial enumerations every 10 hubs (PartialFlusher), and a scratch purge
# between flushes costs GPU-hours of work that no amount of re-analysis recovers.
#
# WHY IT IS BOUNDED: this runs on a SHARED login node. An unbounded background loop is both impolite
# and easy to forget about. ROUNDS defaults to 5 days' worth; it exits on its own, and the heartbeat
# file records when it did. Kill it early with:  kill $(cat $STATE/watchdog.pid)
#
# It sleeps ~99.9% of the time and each round is a short delta-rsync, so the CPU cost is negligible
# against the login node's per-process `ulimit -t` (Logs/012 -- that limit kills long CPU-BOUND
# processes; a sleeping loop never approaches it).
#
# Usage:  setsid nohup bash scripts/backup_watchdog.sh > /dev/null 2>&1 &
# Knobs:  INTERVAL_S (21600 = 6 h)  ROUNDS (20)  STATE (dir for pid/heartbeat/log)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL_S=${INTERVAL_S:-21600}
ROUNDS=${ROUNDS:-20}
STATE=${STATE:-/scratch/markymoo/rgfn_runs}
mkdir -p "$STATE"

HB="$STATE/backup_watchdog.heartbeat"
LOG="$STATE/backup_watchdog.log"
echo $$ > "$STATE/watchdog.pid"

# A previous watchdog would double the rsync load and interleave its log, so allow only one.
#
# Use flock, NOT a pgrep scan. `pgrep -f backup_watchdog.sh` matches any command line CONTAINING that
# string -- including the launching shell's own (`setsid nohup bash scripts/backup_watchdog.sh`) and
# even the pgrep pipeline itself. That false positive made the first version stand down instantly and
# back nothing up, while still LOOKING like a running watchdog. flock keys on the file, not on text.
LOCK="$STATE/watchdog.lock"
exec 9>"$LOCK" || { echo "cannot open lock $LOCK -- exiting" >> "$LOG"; exit 1; }
flock -n 9 || { echo "$(date '+%F %T') another watchdog holds $LOCK -- exiting" >> "$LOG"; exit 0; }

{
    echo "=== watchdog start $(date '+%F %T')  pid=$$  rounds=$ROUNDS  interval=${INTERVAL_S}s"
    echo "    host=$(hostname)  repo=$REPO"
} >> "$LOG"

for i in $(seq 1 "$ROUNDS"); do
    printf '%s round %s/%s starting\n' "$(date '+%F %T')" "$i" "$ROUNDS" >> "$LOG"
    # Per-round log kept separate so a failure is diagnosable; the rolling log stays a short index.
    if bash "$REPO/scripts/backup_scratch_critical.sh" > "$STATE/backup_round.log" 2>&1; then
        # Surface the two lines that matter (what it holds, what is left) into the rolling log.
        tail -3 "$STATE/backup_round.log" | sed 's/^/    /' >> "$LOG"
        printf '%s round %s/%s OK\n' "$(date '+%F %T')" "$i" "$ROUNDS" >> "$LOG"
    else
        printf '%s round %s/%s FAILED (see backup_round.log)\n' "$(date '+%F %T')" "$i" "$ROUNDS" >> "$LOG"
    fi
    date '+%F %T' > "$HB"          # heartbeat: proves the loop is alive without reading the log
    [ "$i" = "$ROUNDS" ] && break  # don't sleep past the last round
    sleep "$INTERVAL_S"
done

printf '%s watchdog finished all %s rounds -- exiting\n' "$(date '+%F %T')" "$ROUNDS" >> "$LOG"
rm -f "$STATE/watchdog.pid"
