#!/usr/bin/env python3
"""Execute the assignment's four parameterized queries, or export both tables."""
import argparse
import json
import math
from pathlib import Path

from semantic_db import add_database_arguments, connect_database, export_database, run_query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_database_arguments(parser)
    sub = parser.add_subparsers(dest="query", required=True)
    nearest = sub.add_parser("nearest")
    nearest.add_argument("class_name")
    nearest.add_argument("x", type=float)
    nearest.add_argument("y", type=float)
    region = sub.add_parser("region")
    for name in ("xmin", "xmax", "ymin", "ymax"):
        region.add_argument(name, type=float)
    sub.add_parser("histogram")
    poses = sub.add_parser("poses")
    poses.add_argument("object_id", type=int)
    export = sub.add_parser("export")
    export.add_argument("--output", type=Path, default=Path("/tmp/semantic-map-export.json"))
    args = parser.parse_args()
    parameters = ()
    if args.query == "nearest":
        if not math.isfinite(args.x) or not math.isfinite(args.y):
            parser.error("coordinates must be finite")
        parameters = (args.x, args.y, args.class_name)
    elif args.query == "region":
        parameters = (args.xmin, args.xmax, args.ymin, args.ymax)
        if (not all(math.isfinite(x) for x in parameters)
                or args.xmin > args.xmax or args.ymin > args.ymax):
            parser.error("invalid rectangle bounds")
    elif args.query == "poses":
        if args.object_id <= 0:
            parser.error("object_id must be positive")
        parameters = (args.object_id,)
    connection = connect_database(args)
    try:
        if args.query == "export":
            result = export_database(connection)
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print(f"Exported {len(result['semantic_objects'])} objects and "
                  f"{len(result['object_observations'])} observations to {args.output}")
        else:
            print(json.dumps(run_query(connection, args.query, parameters), indent=2, allow_nan=False))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
