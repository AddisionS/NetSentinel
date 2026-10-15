
from collections import defaultdict, deque
from datetime import datetime
from threading import Lock, Thread
from scapy.all import sniff, IP, TCP, UDP
from pathlib import Path
import json
import time

WINDOW = 10
IDLE_TIMEOUT = 15

LOG_FILE = Path("live_flows.jsonl")
ALERT_FILE = Path("alerts.jsonl")

flows = {}
recent_syn = defaultdict(deque)
lock = Lock()


def packet_handler(packet):
    if IP not in packet:
        return

    ip = packet[IP]
    src, dst = ip.src, ip.dst

    if TCP in packet:
        protocol = "TCP"
        sport, dport = packet[TCP].sport, packet[TCP].dport
        flags = int(packet[TCP].flags)
        syn = int(bool(flags & 0x02) and not bool(flags & 0x10))
    elif UDP in packet:
        protocol = "UDP"
        sport, dport = packet[UDP].sport, packet[UDP].dport
        syn = 0
    else:
        protocol = "OTHER"
        sport, dport, syn = 0, 0, 0

    endpoints = sorted([(src, sport), (dst, dport)])
    key = (protocol, endpoints[0], endpoints[1])
    now = time.time()

    with lock:
        flow = flows.get(key)

        if flow is None:
            flow = {
                "src_ip": src,
                "dst_ip": dst,
                "src_port": sport,
                "dst_port": dport,
                "protocol": protocol,
                "packets": 0,
                "bytes": 0,
                "syn_count": 0,
                "start": now,
                "last_seen": now,
            }
            flows[key] = flow

        flow["packets"] += 1
        flow["bytes"] += len(packet)
        flow["syn_count"] += syn
        flow["last_seen"] = now

        if syn:
            attempts = recent_syn[src]
            attempts.append(now)

            while attempts and now - attempts[0] > WINDOW:
                attempts.popleft()


def write_record(flow, now):
    duration = max(now - flow["start"], 0.001)
    pps = flow["packets"] / duration
    bps = flow["bytes"] / duration

    attempts = recent_syn[flow["src_ip"]]

    while attempts and now - attempts[0] > WINDOW:
        attempts.popleft()

    alerts = []

    if len(attempts) >= 30:
        alerts.append("POSSIBLE_SYN_SCAN")

    if pps >= 100:
        alerts.append("TRAFFIC_BURST")

    if bps >= 1_000_000:
        alerts.append("HIGH_BYTE_RATE")

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "src_ip": flow["src_ip"],
        "dst_ip": flow["dst_ip"],
        "src_port": flow["src_port"],
        "dst_port": flow["dst_port"],
        "protocol": flow["protocol"],
        "packets": flow["packets"],
        "bytes": flow["bytes"],
        "duration_sec": round(duration, 3),
        "packets_per_sec": round(pps, 2),
        "bytes_per_sec": round(bps, 2),
        "syn_count": flow["syn_count"],
        "alerts": alerts,
    }

    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    for alert in alerts:
        if alert == "TRAFFIC_BURST":
            reason = (
                f"{pps:.2f} packets/sec; "
                "threshold is 100 packets/sec"
            )
        elif alert == "HIGH_BYTE_RATE":
            reason = (
                f"{bps:.2f} bytes/sec; "
                "threshold is 1,000,000 bytes/sec"
            )
        else:
            reason = (
                f"{len(attempts)} TCP SYN attempts "
                f"within {WINDOW} seconds"
            )

        alert_record = {
            **record,
            "alert": alert,
            "severity": "medium",
            "reason": reason,
        }

        with ALERT_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(alert_record) + "\n")

        print(
            f"[ALERT] {alert} | "
            f"{flow['src_ip']} -> {flow['dst_ip']} | "
            f"{reason}"
        )

    print(
        f"[FLOW] {flow['protocol']} | "
        f"{flow['packets']} packets | "
        f"{flow['bytes']} bytes | "
        f"{pps:.2f} packets/sec"
    )


def flush_worker():
    while True:
        time.sleep(1)
        now = time.time()

        with lock:
            ready = []

            for key, flow in list(flows.items()):
                age = now - flow["start"]
                idle = now - flow["last_seen"]

                if age >= WINDOW or idle >= IDLE_TIMEOUT:
                    ready.append(flow)
                    del flows[key]

            for flow in ready:
                write_record(flow, now)


def main():
    print("NetSentinel live capture starting.")
    print(f"Flow log: {LOG_FILE}")
    print(f"Alert log: {ALERT_FILE}")
    print("Stop with Ctrl+C.")

    Thread(target=flush_worker, daemon=True).start()

    try:
        sniff(filter="ip", prn=packet_handler, store=False)

    except PermissionError:
        print("Permission denied. Try PowerShell as Administrator.")

    except OSError as exc:
        print(
            f"Capture error: {exc}. "
            "Check Npcap and capture permissions."
        )

    finally:
        with lock:
            remaining = list(flows.values())
            flows.clear()

            for flow in remaining:
                write_record(flow, time.time())


if __name__ == "__main__":
    main()
