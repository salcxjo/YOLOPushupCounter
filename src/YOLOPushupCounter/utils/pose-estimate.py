import threading
import cv2
from ultralytics import YOLO
from zmq.green import device
import torch
torch.cuda.is_available()

# 1. Load the lightweight YOLOv8 Nano Pose model
# (Automatically downloads on first run)
model = YOLO("yolo26m-pose.pt")


# 2. Optimized Camera Class using Multi-Threading
# This prevents the video from lagging behind real life
class WebcamStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src,cv2.CAP_DSHOW)
        # Force a lower resolution to massively boost processing speed
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
                break
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


# Initialize and start the background thread for the camera
video_feed = WebcamStream(src=0).start()
print("Starting pose tracking... Press 'q' to quit.")

while True:
    frame = video_feed.read()
    if frame is None:
        break

    # 3. Track poses with the tracking engine enabled
    # persist=True maintains person IDs across frames
    # verbose=False stops heavy print statements from slowing down the loop
    results = model.track(frame, stream=True, persist=True, verbose=False)

    # 4. Render the skeletons and bounding boxes
    annotated_frame = frame.copy()
    for result in results:
        annotated_frame = result.plot()

        # OPTIONAL: Access raw keypoint coordinates for your custom logic
        if result.keypoints is not None:
            # keypoints.xy contains [x, y] coordinates for all 17 human joints
            xy_coordinates = result.keypoints.xy.cpu().numpy()
            # print(xy_coordinates) # Uncomment to see the raw array data

    # 5. Display the feed
    cv2.imshow("YOLO Live Pose Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_feed.stop()
cv2.destroyAllWindows()
