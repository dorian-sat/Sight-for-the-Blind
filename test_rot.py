import cv2
import numpy as np
from picamera2 import Picamera2
import pyttsx3
import lgpio
import time

# Text to speech
engine = pyttsx3.init()
engine.setProperty('volume', 1.0)
engine.setProperty('rate', 250)
SCALING_FACTOR = 4

class BlindAssistanceSystem:
    def __init__(self):
        # Initialize lgpio
        self.h = lgpio.gpiochip_open(0)
        
        # Camera initialization
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(main={"size": (int(1232/SCALING_FACTOR), int(1640/SCALING_FACTOR)), "format": "YUV420"})
        self.picam2.configure(config)
        
        # Define zones
        self.width = int(1640/SCALING_FACTOR)
        self.zone_width = self.width // 3
        self.zones = {
            'left': (0, self.zone_width),
            'center': (self.zone_width, self.zone_width * 2),
            'right': (self.zone_width * 2, self.width)
        }
        
        # Ultrasonic sensor pins (TRIG, ECHO)
        self.sensors = {
            'left': {'TRIG': 23, 'ECHO': 24},
            'center': {'TRIG': 17, 'ECHO': 27},
            'right': {'TRIG': 5, 'ECHO': 6}
        }
        
        # Initialize GPIO pins
        for sensor in self.sensors.values():
            lgpio.gpio_claim_output(self.h, sensor['TRIG'])
            lgpio.gpio_claim_input(self.h, sensor['ECHO'])
            lgpio.gpio_write(self.h, sensor['TRIG'], 0)
        
        # Rest of initialization (YOLO, text-to-speech) remains the same
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        weights_path = "yolov3.weights"
        config_path = "yolov3.cfg"
        self.net = cv2.dnn.readNet(weights_path, config_path)
        
        self.classes = []
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        self.last_announcement = {}
        self.min_announcement_interval = 3

    def get_distance(self, zone):
        sensor = self.sensors[zone]
        
        # Trigger measurement
        lgpio.gpio_write(self.h, sensor['TRIG'], 1)
        time.sleep(0.00001)
        lgpio.gpio_write(self.h, sensor['TRIG'], 0)
        
        # Wait for echo
        start_time = time.time()
        while lgpio.gpio_read(self.h, sensor['ECHO']) == 0:
            if time.time() - start_time > 0.1:  # Timeout after 100ms
                return -1
            pulse_start = time.time()
            
        while lgpio.gpio_read(self.h, sensor['ECHO']) == 1:
            if time.time() - start_time > 0.1:  # Timeout after 100ms
                return -1
            pulse_end = time.time()
            
        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150  # Speed of sound * time / 2
        return round(distance, 2)

    # Rest of the methods remain the same
    def get_zone(self, x):
        for zone, (start, end) in self.zones.items():
            if start <= x < end:
                return zone
        return None

    def detect_objects(self, frame):
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        class_ids = []
        confidences = []
        boxes = []
        centers = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    centers.append(center_x)
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        detected_objects = []
        
        # Draw zone lines
        for x in [self.zone_width, self.zone_width * 2]:
            cv2.line(frame, (x, 0), (x, height), (255, 0, 0), 2)
        
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                zone = self.get_zone(centers[i])
                
                if zone:
                    distance = self.get_distance(zone)
                    detection_info = {
                        'zone': zone,
                        'label': label,
                        'distance': distance
                    }
                    detected_objects.append(detection_info)
                    
                    # Print detection details
                    distance = round(distance * 0.0328084, 2)
                    print(f"Object: {label}, Zone: {zone}, Distance: {distance}ft")
                    engine.say(f"{label}, {zone}, {distance} feet")
                    engine.runAndWait()
                    
                    # Draw detection box and label
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ({zone})", (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame, detected_objects

    def announce_objects(self, detected_objects):
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        
        for obj in detected_objects:
            key = f"{obj['zone']}_{obj['label']}"
            if (key not in self.last_announcement or 
                current_time - self.last_announcement[key] >= self.min_announcement_interval):
                self.engine.say(f"{obj['label']}, {obj['zone']}, {obj['distance']} feet")
                self.last_announcement[key] = current_time
        
        self.engine.runAndWait()

    def run(self):
        self.picam2.start()
        
        try:
            while True:
                frame = self.picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
                
                # Rotate frame 90 degrees clockwise to correct orientation
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                
                frame, detected_objects = self.detect_objects(frame)
                #self.announce_objects(detected_objects)
                
                cv2.imshow("Blind Assistance System", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.picam2.stop()
            cv2.destroyAllWindows()
            lgpio.gpiochip_close(self.h)

if __name__ == "__main__":
    system = BlindAssistanceSystem()
    system.run()
