# NEED TO source /opt/ros/humble/setup.bash
# NEED TO source vicon_receiver/install/setup.bash
# reading from bag file tutorial: https://docs.ros.org/en/rolling/Tutorials/Advanced/Reading-From-A-Bag-File-Python.html

import os
import argparse
import csv

import rosbag2_py # Gives you the bytes
import rclpy # teaches Python how to interpret those bytes as ROS messages
from rclpy.serialization import deserialize_message # Serialization = turning a structured message into bytes, Deserialization = turning bytes back into a structured message
from vicon_receiver.msg import Markers # sourcing vicon_receiver allows terminal to see this


def extract_markers_to_csv(bag_path, output_csv):
    # Initialize rclpy (needed for message deserialization)
    rclpy.init()

    # Set up rosbag2 reader
    storage_options = rosbag2_py.StorageOptions( # how to find bag + storage data type
        uri=bag_path,
        storage_id='sqlite3' # "sqlite3" (for .db3)
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )

    reader = rosbag2_py.SequentialReader() # reader to iterate messages in time order
    reader.open(storage_options, converter_options)

    # Optional: inspect topics to make sure topic exists
    topics_and_types = reader.get_all_topics_and_types()
    topic_name = '/vicon/alex_box/markers' # replace later as a list of wanted topics

    topic_found = any(t.name == topic_name for t in topics_and_types)
    if not topic_found:
        print(f"Topic {topic_name} not found in bag. Topics are:")
        for t in topics_and_types:
            print(f"  {t.name} ({t.type})")
        return

    print(f"Found {topic_name}")

    # Create a mapping from topic name to type for quick lookup
    topic_type_map = {t.name: t.type for t in topics_and_types}

    print(topic_type_map)

    # Prepare CSV
    with open(output_csv, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
    
        msg_count = 0
        marker_pos = {}

        # Iterate over all messages
        while reader.has_next():
            # reads one message record at a time where each 
            # message record is (topic_name, serialized_bytes, timestamp_ns)
            # Many messages can share the same time stamp, but rosbag is not stored as frames
            # it's stored as a flat list of message records
            (topic, data, t) = reader.read_next()

            if topic != topic_name:
                continue

            # Deserialize TFMessage
            msg = deserialize_message(data, Markers)

            # Bag timestamp t is in nanoseconds
            t_sec = t * 1e-9

            msg_count += 1

            for marker in msg.markers:
                name = marker.marker_name
                if name not in marker_pos:
                    marker_pos[name] = {
                        'x': marker.translation_x,
                        'y': marker.translation_y,
                        'z': marker.translation_z
                    }

                marker_pos[name]['x'] += marker.translation_x
                marker_pos[name]['y'] += marker.translation_y
                marker_pos[name]['z'] += marker.translation_z

        headers = [f"{marker.marker_name}_{axis}" for marker in msg.markers for axis in ('x', 'y', 'z')]
        writer.writerow(headers)

        marker_pos_avg = []
        for marker in msg.markers:
            name = marker.marker_name
            marker_pos[name]['x'] /= msg_count
            marker_pos[name]['y'] /= msg_count
            marker_pos[name]['z'] /= msg_count

            marker_pos_avg.append(marker_pos[name]['x'])
            marker_pos_avg.append(marker_pos[name]['y'])
            marker_pos_avg.append(marker_pos[name]['z'])

        writer.writerow(marker_pos_avg)

    rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Extract Vicon markers from rosbag2 into a CSV."
    )
    parser.add_argument(
        '--bag',
        required=True,
        help="Path to rosbag2 directory (the one containing metadata.yaml and .db)."
    )
    parser.add_argument(
        '--out',
        default='vicon_markers.csv',
        help="Output CSV file (default: vicon_poses.csv)."
    )
    args = parser.parse_args()

    bag_path = os.path.expanduser(args.bag)
    if not os.path.isdir(bag_path):
        print(f"Error: {bag_path} is not a directory or doesn't exist.")
        return

    extract_markers_to_csv(
        bag_path=bag_path,
        output_csv=args.out
    )


if __name__ == '__main__':
    main()