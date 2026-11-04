import cv2

# 1. Initialize the default webcam (0 is usually the built-in camera)
cap = cv2.VideoCapture(0)

# Check if the webcam opened correctly
if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

# 2. Get the default frame width, height, and FPS from the camera
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20.0  # Standard frame rate for output stability

# 3. Define the video codec and create the VideoWriter object
# 'mp4v' works natively across Windows, Mac, and Linux
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('webcam_output.mp4', fourcc, fps, (frame_width, frame_height))

print("Recording... Press 'q' in the window to stop.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Break the loop if the frame wasn't grabbed successfully
    if not ret:
        print("Error: Failed to grab a frame.")
        break

    # Write the frame to the output file
    out.write(frame)

    # Display the recording live in a window
    cv2.imshow('Recording Webcam... (Press Q to Save)', frame)

    # Wait for 1 millisecond and check if 'q' was pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 4. Clean up and free resources
cap.release()
out.release()
cv2.destroyAllWindows()
print("Video saved successfully as 'webcam_output.mp4'")
