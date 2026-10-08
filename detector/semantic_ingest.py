#!/usr/bin/env python3
"""Receive tb/detections through Zenoh and preserve semantic object histories."""
import argparse
from collections import Counter
import json
from pathlib import Path
import queue
import signal
import threading
import time

from semantic_core import parse_envelope
from semantic_db import ObjectStore, add_database_arguments, connect_database, initialize_database


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    add_database_arguments(parser)
    parser.add_argument("--connect", default="tcp/localhost:7447")
    parser.add_argument("--key", default="tb/detections")
    parser.add_argument("--radius", type=float, default=0.35,
                        help="provisional 3D association radius in metres; measure before reporting")
    parser.add_argument("--record", default="/tmp/semantic-keyframes.jsonl",
                        help="append all valid received keyframes, including empty ones")
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--replay", type=Path, help="ingest an existing keyframe JSONL instead of live Zenoh")
    args = parser.parse_args()
    import math
    if not math.isfinite(args.radius) or args.radius <= 0:
        parser.error("radius must be finite and positive")
    if args.replay and Path(args.record).resolve() == args.replay.resolve():
        parser.error("record and replay must be different files")
    return args


def persist_observation(args, store, observation, stop):
    """Keep the current detection on a transient disconnect; unique IDs make retries safe."""
    import psycopg2
    delay = 1.0
    while not stop.is_set():
        try:
            return store, store.ingest(observation)
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            print("Database disconnected; keeping this observation and retrying.", flush=True)
            try:
                store.connection.close()
            except Exception:
                pass
            if stop.wait(delay):
                break
            delay = min(delay * 2, 5.0)
            try:
                store = ObjectStore(connect_database(args), args.radius)
            except (psycopg2.OperationalError, psycopg2.InterfaceError):
                continue
    raise InterruptedError("ingestion stopped; recorded frames can be replayed")


def main():
    args = arguments()
    connection = connect_database(args)
    session = subscriber = None
    journal = None
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    pending = queue.SimpleQueue()
    counts = Counter()
    store = ObjectStore(connection, args.radius)
    try:
        initialize_database(connection)
        print(f"Database ready: {args.db_host}:{args.db_port}/{args.db_name}", flush=True)
        if args.init_only:
            print("semantic_objects and object_observations are ready.", flush=True)
            return
        journal = open(args.record, "a", encoding="utf-8", buffering=1)
        print(f"Association radius: {args.radius:g} m (provisional; requires measured justification).", flush=True)
        if args.replay:
            with args.replay.open(encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        pending.put(line.encode())
            print(f"Replaying {pending.qsize()} recorded keyframes.", flush=True)
        else:
            import zenoh
            config = zenoh.Config()
            config.insert_json5("connect/endpoints", json.dumps([args.connect]))
            session = zenoh.open(config)
            subscriber = session.declare_subscriber(args.key, lambda sample: pending.put(bytes(sample.payload)))
            print(f"Listening on {args.key}; recording keyframes to {args.record}", flush=True)
            print("Move the robot to produce another keyframe. Ctrl+C stops ingestion.", flush=True)
            # A router storage may have cached data; IDs prevent double insertion.
            try:
                for reply in session.get(args.key, timeout=2.0):
                    try:
                        if reply.ok is not None:
                            pending.put(bytes(reply.ok.payload))
                    except Exception:
                        continue
            except Exception:
                pass
        last_status = time.monotonic()
        while not stop.is_set():
            try:
                payload = pending.get(timeout=0.5)
            except queue.Empty:
                if args.replay:
                    break
                if time.monotonic() - last_status >= 10:
                    print("Ingest status: " + json.dumps(dict(counts)), flush=True)
                    last_status = time.monotonic()
                continue
            try:
                message = json.loads(payload)
                observations = parse_envelope(message)
            except (ValueError, TypeError, KeyError) as error:
                counts["invalid_messages"] += 1
                print(f"Rejected message: {error}", flush=True)
                continue
            # Write before database work so a crash/disconnect can be recovered.
            journal.write(json.dumps(message, allow_nan=False, separators=(",", ":")) + "\n")
            counts["received_keyframes"] += 1
            for observation in observations:
                store, result = persist_observation(args, store, observation, stop)
                counts[result["action"]] += 1
                if result["action"] != "duplicate":
                    print(f"KF#{observation.keyframe_id} {observation.class_name}: "
                          f"object #{result['object_id']} {result['action']} "
                          f"({result.get('candidates', 0)} candidates)", flush=True)
            print(f"KF#{message['keyframe_id']}: {len(observations)} observations; "
                  f"totals {dict(counts)}; queued={pending.qsize()}", flush=True)
        print("Final ingest status: " + json.dumps(dict(counts)), flush=True)
    except InterruptedError as error:
        print(str(error), flush=True)
    finally:
        if subscriber is not None:
            subscriber.undeclare()
        if session is not None:
            session.close()
        if journal is not None:
            journal.close()
        store.connection.close()
        if connection is not store.connection:
            connection.close()


if __name__ == "__main__":
    main()
