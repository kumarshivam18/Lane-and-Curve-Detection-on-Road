# Lane Detection Using OpenCV

This project demonstrates a lane detection system using OpenCV. The system processes a video to detect lane lines and classify road conditions as straight, left curve, right curve, or lane change.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Acknowledgements](#acknowledgements)

## Installation

### Prerequisites

Ensure you have Python 3.x installed. You can download it from [python.org](https://www.python.org/).

### Dependencies

Install the required packages using pip:

```sh
pip install numpy opencv-python
```

## Usage

1. **Place the input video file `lane4.mp4` in the project directory.**

2. **Run the script:**

   ```sh
   python main.py
   ```

## Project Structure

- `main.py`: Main script for processing the video and detecting lanes.
- `README.md`: Project documentation.
- `lane4.mp4`: Input video file (make sure to place it in the project directory).

## Methodology

### 1. Import Libraries

Essential libraries are imported for image processing and video handling.

```python
import cv2
import numpy as np
import os
```

### 2. Define Helper Functions

#### `region_of_interest(img, vertices)`

Creates a mask to focus on the region of interest where lane lines are likely to be.

```python
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked
```

#### `draw_lines(img, lines)`

Draws detected lines on the image.

```python
def draw_lines(img, lines):
    if lines is None:
        return img
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img
```

### 3. Process Function

Processes each frame of the video to detect lanes and classify road conditions.

```python
def process(image):
    height, width = image.shape[:2]
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height)
    ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 200)
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))

    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )

    image_with_lines = draw_lines(image, lines)

    if lines is not None:
        slopes = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
                slopes.append(slope)
        
        left_lane = [slope for slope in slopes if slope < -0.5]
        right_lane = [slope for slope in slopes if slope > 0.5]
        
        if len(left_lane) > 0 and len(right_lane) > 0:
            cv2.putText(image_with_lines, 'Straight', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        elif len(left_lane) > 0:
            cv2.putText(image_with_lines, 'Left Curve', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        elif len(right_lane) > 0:
            cv2.putText(image_with_lines, 'Right Curve', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(image_with_lines, 'Lane Change', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return image_with_lines
```

### 4. Main Function

Handles video input and output, and applies the process function to each frame.

```python
video_path = 'lane4.mp4'

if not os.path.exists(video_path):
    print(f"Error: The file '{video_path}' does not exist.")
else:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'.")
    else:
        print(f"Successfully opened video '{video_path}'.")
        screen_res = 1280, 720  
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Reached the end of the video or cannot read the frame.")
                break

            frame = cv2.resize(frame, screen_res)
            frame = process(frame)
            cv2.imshow('Lane Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
```

## Acknowledgements

This project uses the OpenCV library for computer vision tasks. Special thanks to the contributors of OpenCV.
