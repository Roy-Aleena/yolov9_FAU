#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import torch
import threading

# Load YOLOv9 model
model = torch.hub.load('ultralytics/yolov9', 'yolov9')

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__("image_subscriber")
        self.image_subscription = self.create_subscription(
            Image,
            "/zed_hl1/z/right_raw/image_raw_color",
            self.image_callback,
            10
        )
        self.bridge = CvBridge()
        self.image = None
        self.lock = threading.Lock()
        self.exit_flag = False
        self.thread = threading.Thread(target=self.process_key_input)
        self.thread.start()

    def image_callback(self, msg):
        with self.lock:
            self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def process_key_input(self):
        while not self.exit_flag:
            key = cv2.waitKey(1)  # Wait for a key press
            if key == ord('c'):  # 'c' key pressed
                with self.lock:
                    if self.image is not None:
                        # Run YOLOv9 on the captured image
                        results = model(self.image)

                        # Save the image and YOLOv9 results
                        count = int(cv2.VideoCapture(0).get(cv2.CAP_PROP_FRAME_COUNT))  # Incremental count
                        image_path = f"/home/highlevel/vision_hl1/src/fau_vision/fau_vision/datasets/image_{count}.png"
                        cv2.imwrite(image_path, self.image)
                        results.save(f"/home/highlevel/vision_hl1/src/fau_vision/fau_vision/datasets/results_{count}")

            elif key == ord('q'):  # 'q' key pressed to exit
                self.exit_flag = True

    def cleanup(self):
        self.exit_flag = True
        self.thread.join()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
