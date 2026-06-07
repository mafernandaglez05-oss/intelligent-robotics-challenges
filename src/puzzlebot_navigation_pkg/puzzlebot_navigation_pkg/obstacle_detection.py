#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Float32, String

import cv2
import numpy as np


class ObstacleDetectorCompressed(Node):

    def __init__(self):
        super().__init__("obstacle_detector_compressed")
        self.get_logger().info("Blue obstacle detector compressed started")


        # SUBSCRIBER
        self.create_subscription(
            CompressedImage,
            "/video_source/compressed",
            self.image_callback,
            10
        )


        # PUBLISHERS PARA CONTROLADOR
        self.pub_detected = self.create_publisher(Bool, "/obstacle_detected", 10)
        self.pub_zone = self.create_publisher(String, "/obstacle_zone", 10)
        self.pub_error = self.create_publisher(Float32, "/obstacle_error", 10)
        self.pub_area = self.create_publisher(Float32, "/obstacle_area", 10)

        # PUBLISHERS VISUALES
        self.pub_debug = self.create_publisher(
            CompressedImage,
            "/obstacle_debug/compressed",
            10
        )

        self.pub_mask = self.create_publisher(
            CompressedImage,
            "/obstacle_mask/compressed",
            10
        )


        # PARÁMETROS AJUSTABLES
        self.min_area = 2500

        # ROI: puedes ajustar qué parte de la cámara mira
        self.roi_top_fraction = 0.03
        self.roi_left_fraction = 0.05
        self.roi_right_fraction = 0.95

        # HSV para azul
        # Si no detecta suficiente azul, baja S o V.
        # Si detecta cosas que no son azules, sube S o ajusta H.
        self.lower_blue = np.array([90, 80, 40])
        self.upper_blue = np.array([130, 255, 255])

        # Filtros de tamaño
        self.min_width = 40
        self.min_height = 40
        self.min_area_ratio = 0.01

    def image_callback(self, msg):


        # DECODIFICAR IMAGEN COMPRESSED
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            self.get_logger().warn("No se pudo decodificar la imagen")
            return

        frame = cv2.resize(frame, (640, 480))
        h, w = frame.shape[:2]


        # ROI
        roi_top = int(h * self.roi_top_fraction)
        roi_bottom = h
        roi_left = int(w * self.roi_left_fraction)
        roi_right = int(w * self.roi_right_fraction)

        roi = frame[roi_top:roi_bottom, roi_left:roi_right]


        # HSV + MÁSCARA AZUL
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(
            hsv,
            self.lower_blue,
            self.upper_blue
        )

        # Limpiar ruido
        kernel = np.ones((7, 7), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
            iterations=1
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        mask = cv2.dilate(
            mask,
            kernel,
            iterations=1
        )


        # CONTORNOS
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        obstacle_detected = False
        obstacle_zone = "none"
        obstacle_error = 0.0
        obstacle_area = 0.0

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            x, y, bw, bh = cv2.boundingRect(largest)

            roi_area = roi.shape[0] * roi.shape[1]
            area_ratio = area / roi_area

            if (
                area > self.min_area
                and bw > self.min_width
                and bh > self.min_height
                and area_ratio > self.min_area_ratio
            ):
                obstacle_detected = True
                obstacle_area = float(area)

                cx = x + bw // 2
                cy = y + bh // 2

                roi_w = roi.shape[1]

                # Error normalizado:
                # -1 = izquierda
                #  0 = centro
                # +1 = derecha
                obstacle_error = (cx - roi_w / 2) / (roi_w / 2)

                if obstacle_error < -0.25:
                    obstacle_zone = "left"
                elif obstacle_error > 0.25:
                    obstacle_zone = "right"
                else:
                    obstacle_zone = "center"

                # Coordenadas al frame completo
                x_full = x + roi_left
                y_full = y + roi_top
                cx_full = cx + roi_left
                cy_full = cy + roi_top

                # Dibujar bounding box del objeto azul completo
                cv2.rectangle(
                    frame,
                    (x_full, y_full),
                    (x_full + bw, y_full + bh),
                    (255, 0, 0),
                    2
                )

                cv2.circle(
                    frame,
                    (cx_full, cy_full),
                    5,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    frame,
                    f"BLUE {obstacle_zone} area={area:.0f}",
                    (x_full, y_full - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )


        # PUBLICAR RESULTADOS
        detected_msg = Bool()
        detected_msg.data = obstacle_detected
        self.pub_detected.publish(detected_msg)

        zone_msg = String()
        zone_msg.data = obstacle_zone
        self.pub_zone.publish(zone_msg)

        error_msg = Float32()
        error_msg.data = float(obstacle_error)
        self.pub_error.publish(error_msg)

        area_msg = Float32()
        area_msg.data = float(obstacle_area)
        self.pub_area.publish(area_msg)


        # DEBUG FRAME
        cv2.rectangle(
            frame,
            (roi_left, roi_top),
            (roi_right, roi_bottom),
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Detected: {obstacle_detected}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if obstacle_detected else (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Zone: {obstacle_zone} Error: {obstacle_error:.2f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # PUBLICAR DEBUG / MASK EN COMPRESSED
        self.publish_compressed_image(
            frame,
            msg.header,
            self.pub_debug
        )

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        self.publish_compressed_image(
            mask_bgr,
            msg.header,
            self.pub_mask
        )

    def publish_compressed_image(self, image, header, publisher):
        success, encoded = cv2.imencode(".jpg", image)

        if not success:
            return

        out_msg = CompressedImage()
        out_msg.header = header
        out_msg.format = "jpeg"
        out_msg.data = encoded.tobytes()

        publisher.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetectorCompressed()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
