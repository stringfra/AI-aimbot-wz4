import numpy as np
import torch
import cv2
import time
import pandas as pd
from ultralytics import YOLO
import mss
import tkinter as tk
from tkinter import Scale, HORIZONTAL
from PIL import Image, ImageTk
from pynput.mouse import Button, Controller
from pynput.keyboard import Key, Listener

class ScreenObjectDetector:
    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.1, display_overlay=True):
        self.conf_threshold = conf_threshold
        self.display_overlay = display_overlay
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Caricamento modello {model_name} su {self.device}")
        self.model = YOLO(model_name)
        self.colors = np.random.uniform(0, 255, size=(100, 3))
        self.sct = mss.mss()
        self.monitor = self.get_center_monitor_region()

    def get_center_monitor_region(self, width_percentage=0.5, height_percentage=0.5):
        monitor = self.sct.monitors[1]
        monitor_width = monitor["width"]
        monitor_height = monitor["height"]
        region_width = int(monitor_width * width_percentage)
        region_height = int(monitor_height * height_percentage)
        left = (monitor_width - region_width) // 2
        top = (monitor_height - region_height) // 2
        return {"top": top, "left": left, "width": region_width, "height": region_height}

    def adjust_capture_area(self, width, height, left=0, top=0):
        self.monitor = {"top": top, "left": left, "width": width, "height": height}

    def capture_screen(self):
        sct_img = self.sct.grab(self.monitor)
        return np.array(sct_img)[:, :, :3]

    def detect_screen(self):
        frame = self.capture_screen()
        results = self.model(frame, verbose=False)
        detections = []
        annotated_frame = frame.copy() if self.display_overlay else None
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                if confidence < self.conf_threshold:
                    continue
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                if class_name in ['head', 'body']:
                    detections.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "center_x": center_x,
                        "center_y": center_y,
                        "width": width,
                        "height": height,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    })
                    if self.display_overlay and annotated_frame is not None:
                        color = self.colors[class_id % len(self.colors)]
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        label = f"{class_name}: {confidence:.2f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(annotated_frame, (int(x1), int(y1) - label_size[1] - 5), (int(x1) + label_size[0], int(y1)), color, -1)
                        cv2.putText(annotated_frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return pd.DataFrame(detections) if detections else pd.DataFrame(), annotated_frame

class OverlayApp:
    def __init__(self, detector):
        self.detector = detector
        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.geometry(f"{self.detector.monitor['width']}x{self.detector.monitor['height']}+{self.detector.monitor['left']}+{self.detector.monitor['top']}")

        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.circle_canvas = tk.Canvas(self.root, highlightthickness=0, relief='ridge')
        self.circle_canvas.place(x=0, y=0, width=self.detector.monitor['width'], height=self.detector.monitor['height'])

        self.running = True
        self.fps = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.mouse = Controller()
        self.keyboard = Controller()
        self.drag_data = {"item": None, "x": 0, "y": 0}
        self.listener = Listener(on_press=self.on_key_press)
        self.listener.start()
        self.detection_interval = 1
        self.frame_count = 0
        self.fov = 70
        self.update_interval = int(1000 / self.fov)

        self.gui_root = tk.Toplevel(self.root)
        self.fov_scale = Scale(self.gui_root, from_=1, to=100, orient=HORIZONTAL, label="FoV", command=self.update_fov)
        self.fov_scale.set(self.fov)
        self.fov_scale.pack()

        self.root.attributes("-alpha", 0.8)
        self.canvas.config(bg='black')
        self.circle_canvas.config(bg="")

        self.update_detection()
        self.root.bind("<Escape>", lambda e: self.quit())

    def update_fov(self, fov):
        self.fov = int(fov)
        self.update_interval = int(1000 / self.fov)
        self.detector.monitor = self.detector.get_center_monitor_region(width_percentage=self.fov/100, height_percentage=self.fov/100)
        self.root.geometry(f"{self.detector.monitor['width']}x{self.detector.monitor['height']}+{self.detector.monitor['left']}+{self.detector.monitor['top']}")
        self.circle_canvas.config(width=self.detector.monitor['width'], height=self.detector.monitor['height'])

    def update_detection(self):
        if not self.running:
            return
        try:
            self.frame_count += 1
            if self.frame_count % self.detection_interval == 0:
                detections, annotated_frame = self.detector.detect_screen()
                self.frame_count = 0
            else:
                annotated_frame = self.detector.capture_screen()
                detections = pd.DataFrame()

            if annotated_frame is not None:
                self.update_fps(annotated_frame)
                self.display_frame(annotated_frame, detections)
                self.draw_fov_circle()

        except Exception as e:
            print(f"Errore durante l'aggiornamento: {e}")

        self.root.after(self.update_interval, self.update_detection)

    def update_fps(self, frame):
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()
        cv2.putText(frame, f"FPS: {self.fps}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    def draw_fov_circle(self):
        self.circle_canvas.delete("fov_circle")
        center_x = self.detector.monitor['width'] // 2
        center_y = self.detector.monitor['height'] // 2
        radius = min(center_x, center_y) * (self.fov / 100)
        self.circle_canvas.create_oval(center_x - radius, center_y - radius, center_x + radius, center_y + radius, outline="white", width=2, tags="fov_circle")

    def on_click(self, event, detections):
        x, y = event.x, event.y
        for _, row in detections.iterrows():
            if row["x1"] < x < row["x2"] and row["y1"] < y < row["y2"]:
                print(f"Oggetto cliccato: {row['class_name']}")
                self.mouse.position = (int(row["center_x"] + self.detector.monitor["left"]), int(row["center_y"] + self.detector.monitor["top"]))
                self.mouse.click(Button.left)
                break

    def on_press(self, event, detections):
        x, y = event.x, event.y
        for _, row in detections.iterrows():
            if row["x1"] < x < row["x2"] and row["y1"] < y < row["y2"]:
                self.drag_data["item"] = row
                self.drag_data["x"] = x
                self.drag_data["y"] = y
                break

    def on_motion(self, event):
        if self.drag_data["item"]:
            delta_x = event.x - self.drag_data["x"]
            delta_y = event.y - self.drag_data["y"]
            self.mouse.move(delta_x, delta_y)
            self.drag_data["x"] = event.x
            self.drag_data["y"] = event.y

    def on_release(self, event):
        self.drag_data["item"] = None
        self.drag_data["x"] = 0
        self.drag_data["y"] = 0

    def on_key_press(self, key):
        try:
            print(f"Tasto premuto: {key.char}")
            if key.char == "a":
                self.keyboard.press("a")
                self.keyboard.release("a")
        except AttributeError:
            print(f"Tasto speciale premuto: {key}")
            if key == Key.esc:
                self.quit()

    def quit(self):
        self.running = False
        self.root.destroy()
        self.listener.stop()


    def main():
        detector = ScreenObjectDetector(model_name="yolov8n.pt", conf_threshold=0.1)
        app = OverlayApp(detector)
        print("Applicazione avviata. Premi ESC per uscire.")
        app.root.mainloop()