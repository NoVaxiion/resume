"""
Live monitors a screenshots folder locally, OCRs new screenshots, and saves detected fields into ocr_labels.csv for manual labeling.
"""

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import pytesseract
import time
import cv2
import csv
import os

#Path where screenshots are saved 
#Make sure to have where screenshot directly goes to a folder
WATCHED_FOLDER = '/Users/novaxion/Desktop/Screenshots'

#Path to save labeled data
OUTPUT_CSV = '../data/ocr_labels.csv'

#Initialize CSV if it doesn't exist
if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["field_text", "target_label"])

class ScreenshotHandler(FileSystemEventHandler):
    def process(self, event):
        if event.is_directory:
            return

        filepath = event.src_path
        filename = os.path.basename(filepath)

        if not filename.startswith('.') and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.heic', '.webp')):
            print(f"New or Modified Screenshot: {filepath}")

            #Wait a bit to ensure file is fully saved
            time.sleep(3)

            img = cv2.imread(filepath)

            if img is None:
                print(f"Warning: Unable to open image {filepath}")
                return

            ocr_result = pytesseract.image_to_string(img)

            fields = [line.strip() for line in ocr_result.split("\n") if line.strip()]
            print(f"Fields Detected: {fields}")

            with open(OUTPUT_CSV, mode='a', newline='') as file:
                writer = csv.writer(file)
                for field in fields:
                    writer.writerow([field, ""])  #Empty target_label for later labeling

            print(f"Saved {len(fields)} fields to {OUTPUT_CSV}")

    def on_created(self, event):
        self.process(event)

    def on_modified(self, event):
        self.process(event)

def main():
    event_handler = ScreenshotHandler()
    observer = Observer()
    observer.schedule(event_handler, path=WATCHED_FOLDER, recursive=False)
    observer.start()
    print(f"Watching {WATCHED_FOLDER} for new screenshots... Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
