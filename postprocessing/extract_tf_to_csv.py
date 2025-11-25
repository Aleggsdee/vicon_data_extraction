#!/usr/bin/env python3
# NEED TO source /opt/ros/humble/setup.bash
import os
import argparse
import csv

import rclpy
from rclpy.serialization import deserialize_message
from tf2_msgs.msg import TFMessage

import rosbag2_py


def extract_tf_to_csv(bag_path, target_child_frame, output_csv):
    # Initialize rclpy (needed for message deserialization)
    rclpy.init()

    # Set up rosbag2 reader
    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id='sqlite3'
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    # Optional: inspect topics to make sure /tf exists
    topics_and_types = reader.get_all_topics_and_types()
    tf_topic_name = '/tf'
    tf_type = 'tf2_msgs/msg/TFMessage'

    topic_found = any(t.name == tf_topic_name for t in topics_and_types)
    if not topic_found:
        print(f"Topic {tf_topic_name} not found in bag. Topics are:")
        for t in topics_and_types:
            print(f"  {t.name} ({t.type})")
        return

    print(f"Found {tf_topic_name}. Extracting transforms with child_frame_id='{target_child_frame}'")

    # Create a mapping from topic name to type for quick lookup
    topic_type_map = {t.name: t.type for t in topics_and_types}

    # Prepare CSV
    with open(output_csv, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['t', 'px', 'py', 'pz', 'qx', 'qy', 'qz', 'qw'])

        msg_count = 0
        matched_count = 0

        # Iterate over all messages
        while reader.has_next():
            (topic, data, t) = reader.read_next()

            if topic != tf_topic_name:
                continue

            # Deserialize TFMessage
            msg = deserialize_message(data, TFMessage)

            # Bag timestamp t is in nanoseconds
            t_sec = t * 1e-9

            for tf in msg.transforms:
                msg_count += 1

                if tf.child_frame_id != target_child_frame:
                    continue

                matched_count += 1

                # You can also use tf.header.stamp, but t from bag is usually fine
                # If you'd rather use header.stamp:
                # stamp = tf.header.stamp
                # t_sec = stamp.sec + stamp.nanosec * 1e-9

                p = tf.transform.translation
                q = tf.transform.rotation

                writer.writerow([
                    t_sec,
                    p.x, p.y, p.z,
                    q.x, q.y, q.z, q.w
                ])

        print(f"Done. Read {msg_count} transforms on {tf_topic_name}.")
        print(f"Wrote {matched_count} rows to {output_csv} "
              f"for child_frame_id='{target_child_frame}'.")

    rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Extract Vicon poses from rosbag2 /tf into a CSV."
    )
    parser.add_argument(
        '--bag',
        required=True,
        help="Path to rosbag2 directory (the one containing metadata.yaml and .db)."
    )
    parser.add_argument(
        '--child-frame',
        required=True,
        default="alex_box",
        help="child_frame_id of the object in TF (e.g. 'box', 'RigidBody1')."
    )
    parser.add_argument(
        '--out',
        default='vicon_poses.csv',
        help="Output CSV file (default: vicon_poses.csv)."
    )
    args = parser.parse_args()

    bag_path = os.path.expanduser(args.bag)
    if not os.path.isdir(bag_path):
        print(f"Error: {bag_path} is not a directory or doesn't exist.")
        return

    extract_tf_to_csv(
        bag_path=bag_path,
        target_child_frame=args.child_frame,
        output_csv=args.out
    )


if __name__ == '__main__':
    main()
