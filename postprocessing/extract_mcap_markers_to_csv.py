#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mcap_path", help="Path to .mcap file")
    ap.add_argument("--topic", default="/vicon/markers")
    ap.add_argument("--out", default="markers.csv")
    args = ap.parse_args()

    mcap_path = Path(args.mcap_path).expanduser().resolve()
    out_csv = Path(args.out).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = DecoderFactory()

    n_msgs = 0
    n_rows = 0

    with mcap_path.open("rb") as f, out_csv.open("w", newline="") as g:
        w = csv.writer(g)
        w.writerow([
            "t_ns",
            "log_time_ns",
            "frame_number",
            "marker_idx_in_msg",
            "index",
            "marker_name",
            "subject_name",
            "segment_name",
            "occluded",
            "x", "y", "z"
        ])

        reader = make_reader(f)

        for schema, channel, raw_msg in reader.iter_messages(topics=[args.topic]):
            dec = df.decoder_for(channel.message_encoding, schema)
            if dec is None:
                continue

            msg = dec(raw_msg.data)
            n_msgs += 1
            log_time_ns = int(raw_msg.log_time)

            try:
                t_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
            except Exception:
                t_ns = log_time_ns

            frame_number = int(getattr(msg, "frame_number", 0))

            for i, m in enumerate(msg.markers):
                w.writerow([
                    t_ns,
                    log_time_ns,
                    frame_number,
                    i,
                    int(m.index),
                    m.marker_name,
                    m.subject_name,
                    m.segment_name,
                    bool(m.occluded),
                    float(m.translation.x),
                    float(m.translation.y),
                    float(m.translation.z),
                ])
                n_rows += 1

    print(f"Wrote {n_rows} marker rows from {n_msgs} messages to: {out_csv}")


if __name__ == "__main__":
    main()
