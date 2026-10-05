#!/usr/bin/env python3
"""Synthetic operations-incident contexts for long-prompt measurements.

Builds a deterministic "incident bundle" of the kind an operations request pastes into a prompt:
journald excerpts from several services, an nginx access log, Kubernetes events, application
stack traces, a deployment manifest and a ConfigMap, shell history and a Prometheus metrics dump.
All content is generated; nothing comes from a real system. Every bundle carries one planted
incident chain:
  a ConfigMap change shrinks a service's database pool, then that service logs pool exhaustion,
  nginx returns 503s for its routes, readiness probes fail and pods restart.
The planted facts (`Bundle.facts`) let a driver check whether an answer used the context: whether
it names the service, the release, the pool setting and the change. They are a smoke signal, not
a quality score.

    ops_corpus.py --chars 60000 --seed 3 > bundle.txt      # inspect one bundle

Tasks (`TASKS`) are the request shapes measured: write test/review/log scripts for the incident,
gather evidence for root-cause analysis and triage, and review an existing collection script
(`build(..., script=True)` adds it to the bundle with three planted bugs) and return it corrected.
The edit task's answer repeats the script almost verbatim.
`FOLLOWUPS` are the second turns that reuse the first turn's prefix.
"""
import argparse
import dataclasses
import random

SERVICES = ["payments-api", "orders-api", "inventory-svc", "auth-gateway", "notify-worker",
            "search-indexer", "billing-cron", "catalog-api", "session-store", "ledger-sync"]
HOSTS = [f"node-{rack}{n:02d}" for rack in "abc" for n in range(1, 7)]
ROUTES = {"payments-api": ["/v2/payments", "/v2/payments/{id}", "/v2/refunds"],
          "orders-api": ["/v1/orders", "/v1/orders/{id}", "/v1/cart"],
          "inventory-svc": ["/v1/stock", "/v1/reservations"],
          "auth-gateway": ["/oauth/token", "/oauth/introspect"],
          "catalog-api": ["/v1/products", "/v1/products/{id}", "/v1/search"]}
NOISE = [
    "GET /healthz 200 {ms}ms",
    "cache hit ratio {pct}% over last 60s",
    "flushed {n} metrics to collector",
    "refreshed JWKS keys (kid={hex})",
    "worker {n} heartbeat ok",
    "processed batch id={hex} records={n} in {ms}ms",
    "slow query {ms}ms: SELECT * FROM {table} WHERE id = $1",
    "retrying upstream call to {svc} (attempt 2/3)",
    "GC pause {ms}ms (young)",
    "rotated log file, size={n}KB",
    "TLS session resumed for client {ip}",
    "config reload skipped: checksum unchanged",
]
TABLES = ["orders", "payments", "ledger_entries", "sessions", "stock_levels", "refunds"]
TASKS = {
    "script": (
        "You are on call. Using only the incident bundle above, write a bash script that (1) "
        "collects the evidence for this incident from journald, the nginx access log and "
        "kubectl into a timestamped directory, (2) runs basic checks that would have caught it "
        "(probe failures, 5xx rate per route, pool or connection errors per service), and (3) "
        "logs every step with timestamps to a file. Then write a short Python script that "
        "parses the nginx log format shown above and prints the 5xx rate per route per minute. "
        "Explain each script's assumptions briefly."),
    "rca": (
        "You are on call. Using only the incident bundle above, gather the evidence for a root "
        "cause analysis and triage this incident: give a timeline with exact timestamps, the "
        "most likely root cause and the evidence for it, the services and routes affected, what "
        "is noise, the immediate mitigation, and the follow-up actions. Quote the log lines you "
        "rely on."),
    "edit": (
        "You are on call. Review the script ops/collect_evidence.sh in the incident bundle above "
        "against the bundle's own output: it should have collected the evidence for this "
        "incident, but it has bugs. Return the complete corrected script in one code block, "
        "keeping everything that is correct unchanged, then list each fix in one line."),
}
FOLLOWUPS = {
    "script": "Now add a check to the bash script that compares the live ConfigMap with the "
              "previous revision and alerts if any pool or timeout setting shrank. Show only the "
              "new function and where it is called.",
    "rca": "Write the incident summary for the status page (four sentences) and a list of the "
           "three log queries an engineer should save for the next occurrence.",
    "edit": "Add a --since option (default 2h) to the corrected script and use it for every "
            "journalctl and kubectl logs call. Return the complete script again.",
}


@dataclasses.dataclass
class Bundle:
    text: str
    facts: dict


def _ts(minute, second, ms=0):
    hour = 9 + minute // 60
    return f"2026-08-14T{hour:02d}:{minute % 60:02d}:{second:02d}.{ms:03d}Z"


def _fill(rng, template, svc):
    return template.format(ms=rng.randint(2, 900), pct=rng.randint(70, 99),
                           n=rng.randint(10, 9000), hex=f"{rng.getrandbits(40):010x}",
                           table=rng.choice(TABLES), svc=svc,
                           ip=f"10.{rng.randint(0, 9)}.{rng.randint(0, 255)}.{rng.randint(2, 254)}")


def collection_script(victim):
    """The on-call team's evidence script with three planted bugs: the namespace `production`
    (the cluster uses `prod`), the ConfigMap name `<service>-cfg` (it is `<service>-config`), and a
    log written with `>` so every step overwrites the previous one."""
    return f"""=== cat ops/collect_evidence.sh ===
#!/usr/bin/env bash
# Collect the evidence for a {victim} incident into a timestamped directory.
# Usage: ops/collect_evidence.sh [output-root]
set -euo pipefail

SERVICE={victim}
NAMESPACE=production
OUT_ROOT=${{1:-/var/tmp/incidents}}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$OUT_ROOT/$SERVICE-$STAMP"
LOG="$OUT/collect.log"
mkdir -p "$OUT"

log() {{
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" > "$LOG"
}}

collect_journal() {{
    log "journald for $SERVICE"
    journalctl --since "-2h" --no-pager -o short-iso | grep -F "$SERVICE[" > "$OUT/journal.txt" || true
    log "journald lines: $(wc -l < "$OUT/journal.txt")"
}}

collect_nginx() {{
    log "nginx 5xx for $SERVICE"
    grep -F "upstream=$SERVICE" /var/log/nginx/access.log > "$OUT/nginx.txt" || true
    awk '$9 >= 500 {{ print }}' "$OUT/nginx.txt" > "$OUT/nginx-5xx.txt"
    log "nginx 5xx lines: $(wc -l < "$OUT/nginx-5xx.txt")"
}}

collect_kubernetes() {{
    log "kubernetes state in $NAMESPACE"
    kubectl -n "$NAMESPACE" get pods -o wide > "$OUT/pods.txt"
    kubectl -n "$NAMESPACE" get events --sort-by=.lastTimestamp > "$OUT/events.txt"
    kubectl -n "$NAMESPACE" get configmap "$SERVICE-cfg" -o yaml > "$OUT/configmap.yaml"
    kubectl -n "$NAMESPACE" rollout history "deployment/$SERVICE" > "$OUT/rollout.txt"
    kubectl -n "$NAMESPACE" logs "deploy/$SERVICE" --since=2h > "$OUT/app.log" || true
}}

check_pool() {{
    log "pool errors"
    grep -c "QueuePool limit" "$OUT/journal.txt" > "$OUT/pool-errors.txt" || true
    log "pool errors: $(cat "$OUT/pool-errors.txt")"
}}

check_probes() {{
    log "readiness probe failures"
    grep -c "Readiness probe failed" "$OUT/events.txt" > "$OUT/probe-failures.txt" || true
    log "probe failures: $(cat "$OUT/probe-failures.txt")"
}}

collect_journal
collect_nginx
collect_kubernetes
check_pool
check_probes
log "done: $OUT"
echo "$OUT"
"""


def build(chars, seed, script=False):
    """One incident bundle of at most `chars` characters, ending on a whole line. With `script`,
    the bundle also carries ops/collect_evidence.sh for the edit task."""
    rng = random.Random(seed)
    victim = rng.choice(list(ROUTES))
    release = f"v{rng.randint(2, 4)}.{rng.randint(1, 30)}.{rng.randint(0, 9)}"
    old_pool, new_pool = rng.choice([(50, 5), (40, 4), (64, 8), (32, 2)])
    change_min = rng.randint(25, 40)          # minute of the ConfigMap rollout
    first_err = change_min + rng.randint(2, 4)
    facts = {"service": victim, "release": release, "old_pool": old_pool, "new_pool": new_pool,
             "change_time": _ts(change_min, 12), "first_error": _ts(first_err, 0)}

    sections = []
    # Deployment manifest and ConfigMap: the planted change.
    sections.append(
        f"=== kubectl -n prod get configmap {victim}-config -o yaml (current) ===\n"
        f"apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: {victim}-config\n  namespace: prod\n"
        f"  annotations:\n    deploy.example/release: {release}\n"
        f"    deploy.example/applied-at: \"{facts['change_time']}\"\ndata:\n"
        f"  DB_HOST: pg-primary.prod.svc\n  DB_POOL_SIZE: \"{new_pool}\"\n  DB_POOL_OVERFLOW: \"0\"\n"
        f"  DB_POOL_TIMEOUT_S: \"30\"\n  HTTP_WORKERS: \"16\"\n  LOG_LEVEL: info\n")
    sections.append(
        f"=== kubectl -n prod rollout history deployment/{victim} ===\nREVISION  CHANGE-CAUSE\n"
        + "".join(f"{r}         release v{rng.randint(1, 2)}.{rng.randint(0, 30)}.{rng.randint(0, 9)}\n"
                  for r in range(31, 36))
        + f"36        release {release} (config: pool tuning)\n")
    sections.append(
        f"=== git log -3 --stat deploy/{victim}/configmap.yaml ===\n"
        f"commit {rng.getrandbits(160):040x}\nDate:   {facts['change_time']}\n\n"
        f"    {victim}: pool tuning for {release}\n\n deploy/{victim}/configmap.yaml | 2 +-\n"
        f"-  DB_POOL_SIZE: \"{old_pool}\"\n+  DB_POOL_SIZE: \"{new_pool}\"\n")
    if script:
        sections.append(collection_script(victim))

    def journald(minutes):
        lines = []
        for minute in minutes:
            for _ in range(rng.randint(6, 12)):
                svc = rng.choice(SERVICES)
                host = rng.choice(HOSTS)
                pid = rng.randint(1000, 60000)
                sec = rng.randint(0, 59)
                if svc == victim and minute >= first_err and rng.random() < 0.55:
                    msg = rng.choice([
                        f"ERROR sqlalchemy.pool: QueuePool limit of size {new_pool} overflow 0 "
                        f"reached, connection timed out, timeout 30.00",
                        "ERROR request failed: TimeoutError acquiring DB connection after 30.0s",
                        f"WARN pool checked out {new_pool}/{new_pool}, {rng.randint(20, 300)} "
                        "requests waiting"])
                elif rng.random() < 0.03:
                    msg = f"WARN upstream {rng.choice(SERVICES)} responded slowly ({rng.randint(900, 2500)}ms)"
                else:
                    msg = "INFO " + _fill(rng, rng.choice(NOISE), rng.choice(SERVICES))
                lines.append((minute, sec, f"{_ts(minute, sec, rng.randint(0, 999))} {host} {svc}[{pid}]: {msg}"))
        lines.sort()
        return "".join(line + "\n" for _, _, line in lines)

    def nginx(minutes):
        lines = []
        for minute in minutes:
            for _ in range(rng.randint(10, 18)):
                svc = rng.choice(list(ROUTES))
                route = rng.choice(ROUTES[svc]).replace("{id}", str(rng.randint(10000, 99999)))
                status = 200
                if svc == victim and minute >= first_err and rng.random() < 0.6:
                    status = rng.choice([503, 503, 504])
                elif rng.random() < 0.01:
                    status = rng.choice([404, 429, 500])
                rt = rng.uniform(30.0, 31.0) if status in (503, 504) else rng.uniform(0.003, 0.4)
                sec = rng.randint(0, 59)
                lines.append((minute, sec,
                              f"10.{rng.randint(0, 9)}.{rng.randint(0, 255)}.{rng.randint(2, 254)} - - "
                              f"[14/Aug/2026:{9 + minute // 60:02d}:{minute % 60:02d}:{sec:02d} +0000] "
                              f"\"{rng.choice(['GET', 'POST', 'GET', 'PUT'])} {route} HTTP/1.1\" {status} "
                              f"{rng.randint(80, 9000)} \"-\" \"svc-client/{rng.randint(1, 4)}.0\" "
                              f"rt={rt:.3f} upstream={svc}"))
        lines.sort()
        return "".join(line + "\n" for _, _, line in lines)

    def events(minutes):
        lines = []
        for minute in minutes:
            if minute >= first_err + 1 and rng.random() < 0.7:
                pod = f"{victim}-{rng.getrandbits(32):08x}-{rng.getrandbits(20):05x}"
                lines.append(f"{_ts(minute, rng.randint(0, 59))} Warning Unhealthy pod/{pod} "
                             "Readiness probe failed: HTTP probe failed with statuscode: 503")
                if rng.random() < 0.4:
                    lines.append(f"{_ts(minute, rng.randint(0, 59))} Warning BackOff pod/{pod} "
                                 "Back-off restarting failed container")
            for _ in range(rng.randint(1, 3)):
                svc = rng.choice(SERVICES)
                lines.append(f"{_ts(minute, rng.randint(0, 59))} Normal "
                             f"{rng.choice(['Pulled', 'Scheduled', 'Created', 'Started'])} "
                             f"pod/{svc}-{rng.getrandbits(32):08x}-{rng.getrandbits(20):05x}")
        return "".join(line + "\n" for line in sorted(lines))

    def traces(count):
        out = []
        for _ in range(count):
            out.append(
                f"--- {victim} {rng.choice(HOSTS)} {_ts(rng.randint(first_err, first_err + 50), rng.randint(0, 59))} ---\n"
                "Traceback (most recent call last):\n"
                f"  File \"/app/{victim.replace('-', '_')}/handlers.py\", line {rng.randint(40, 400)}, in handle\n"
                "    with session_scope() as session:\n"
                "  File \"/app/common/db.py\", line 88, in session_scope\n"
                "    conn = engine.connect()\n"
                "  File \"/usr/lib/python3.11/site-packages/sqlalchemy/pool/impl.py\", line 168, in _do_get\n"
                f"sqlalchemy.exc.TimeoutError: QueuePool limit of size {new_pool} overflow 0 reached, "
                "connection timed out, timeout 30.00\n")
            if rng.random() < 0.5:
                other = rng.choice([s for s in SERVICES if s != victim])
                out.append(
                    f"--- {other} {rng.choice(HOSTS)} {_ts(rng.randint(0, 99), rng.randint(0, 59))} ---\n"
                    "java.net.SocketTimeoutException: Read timed out\n"
                    f"\tat com.example.{other.replace('-', '')}.client.HttpClient.call(HttpClient.java:{rng.randint(50, 300)})\n"
                    "\tat com.example.common.Retry.run(Retry.java:41)\n")
        return "".join(out)

    def shell(count):
        cmds = ["kubectl -n prod get pods -o wide", "kubectl -n prod top pods",
                "journalctl -u kubelet --since -10m", "df -h /var/lib/containerd",
                "kubectl -n prod describe pod {svc}-7f9c", "curl -s localhost:9090/-/healthy",
                "kubectl -n prod logs deploy/{svc} --since=5m | tail -50", "uptime",
                "psql -h pg-primary.prod.svc -c 'select count(*) from pg_stat_activity'"]
        return "".join(f"{rng.randint(1000, 9999)}  {_ts(rng.randint(0, 99), rng.randint(0, 59))}  "
                       f"{rng.choice(cmds).format(svc=rng.choice(SERVICES))}\n" for _ in range(count))

    def metrics(minutes):
        out = []
        for minute in minutes:
            for svc in SERVICES:
                busy = new_pool if (svc == victim and minute >= first_err) else rng.randint(1, 12)
                out.append(f"db_pool_connections_in_use{{service=\"{svc}\"}} {busy} {_ts(minute, 0)}\n")
            out.append(f"pg_stat_activity_count{{db=\"prod\"}} {rng.randint(80, 140)} {_ts(minute, 0)}\n")
        return "".join(out)

    window = list(range(0, 100))
    generators = [
        lambda block: "=== journalctl --since 09:00 (selected services) ===\n" + journald(block),
        lambda block: "=== /var/log/nginx/access.log (edge-lb-1) ===\n" + nginx(block),
        lambda block: "=== kubectl -n prod get events --sort-by=.lastTimestamp ===\n" + events(block),
        lambda block: "=== application stack traces (sampled) ===\n" + traces(3),
        lambda block: "=== ~/.bash_history (on-call, timestamped) ===\n" + shell(12),
        lambda block: "=== prometheus export (1m resolution) ===\n" + metrics(block),
    ]
    header = ("Incident bundle INC-%05d, collected by the on-call engineer. Sections are raw tool "
              "output.\n\n" % rng.randint(0, 99999))
    budget = chars - len(header) - sum(len(section) for section in sections)
    chunks = []
    step = 0
    while sum(len(chunk) for chunk in chunks) < budget:
        start = (step * 7) % 90
        chunks.append(generators[step % len(generators)](window[start:start + 10]))
        step += 1
    # Trim the last section at a line boundary to fit, then put the planted change 40-70% of the
    # way in, so an answer that names it has read deep into the context.
    over = sum(len(chunk) for chunk in chunks) - budget
    if over > 0:
        kept = chunks[-1][:len(chunks[-1]) - over]
        chunks[-1] = kept[:kept.rfind("\n") + 1]
    at = int(len(chunks) * rng.uniform(0.4, 0.7))
    text = "".join(chunks[:at] + sections + chunks[at:])
    return Bundle(header + text, facts)


def mentions(answer, facts):
    """Which planted facts an answer names (case-insensitive substring checks)."""
    lower = answer.lower()
    return {
        "service": facts["service"] in lower,
        "release": facts["release"].lower() in lower,
        "pool": "pool" in lower and str(facts["new_pool"]) in lower,
        "change": "configmap" in lower or ("config" in lower and str(facts["old_pool"]) in lower),
    }


def script_fixes(answer, facts):
    """Which of collection_script's three planted bugs an edit answer fixed."""
    return {
        "namespace": "NAMESPACE=prod\n" in answer,
        "configmap": f'"$SERVICE-config"' in answer or f"{facts['service']}-config" in answer,
        "log_append": '>> "$LOG"' in answer,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chars", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--script", action="store_true", help="include ops/collect_evidence.sh")
    args = ap.parse_args()
    bundle = build(args.chars, args.seed, args.script)
    print(bundle.text)
    print(f"\n# facts: {bundle.facts}")


if __name__ == "__main__":
    main()
