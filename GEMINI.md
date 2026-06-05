# Project Context: Arrow Graph Analysis

## Overview
This Python project specializes in computer vision tasks to detect, analyze, and convert "arrow puzzle" diagrams into structured graph data. It processes images (e.g., game screenshots) to identify arrows, determine their positions and orientations, and reconstruct the logical grid or graph they represent.

## Architecture
The system is built on **OpenCV (`cv2`)** and **NumPy**. It follows a pipeline approach:
1.  **Preprocessing:** Images are cropped to remove UI elements and focused on the puzzle area.
2.  **Detection:** Contours are extracted from binary thresholded images.
3.  **Analysis:** Geometric algorithms identify arrow heads, centers, and skeletal paths of the connecting lines.

## Key Files
*   **`arrow_graph_converter.py`**: The primary entry point for analyzing a puzzle image. It orchestrates the cropping, thresholding, and contour processing.
*   **`smart_cropper.py`**: Contains logic to intelligently crop input images. It specifically filters out top/bottom UI bars to isolate the central puzzle grid.
*   **`arrow_utils.py`**: A utility module for geometric calculations. Includes `get_arrow_head_center` which uses convexity and angle analysis to find the tip of an arrow shape.
*   **`tests/`**:
    *   `visualize_arrow_centers.py`: Runs the detection logic on sample images and saves visual overlays (red dots on arrow tips) to verify accuracy.
    *   `draw_lines_test.py`: Experiments with skeletonization and path pruning to trace the lines connecting arrows.

## Setup & Usage

### Dependencies
The project relies on a standard Python environment.
*   **Virtual Environment:** A `.venv` directory is present. Ensure it is activated.
*   **Packages:** Major dependencies include `opencv-python` and `numpy`.

### Running Tests
The `tests` folder contains executable scripts that process images from `sample_ui` and output results to the `tests` directory.

```bash
# Example: Visualize arrow detection
python tests/visualize_arrow_centers.py

# Example: Test line drawing/skeletonization
python tests/draw_lines_test.py
```

## Conventions
*   **Image Processing:** Images are generally converted to grayscale and then binary thresholded (inverted) so that dark arrows become white regions on a black background for contour detection.
*   **Testing:** Tests are visual. They generate output images (e.g., `result_level2.png`) which must be inspected manually to verify correctness.
*   **Path Handling:** Scripts typically add the project root to `sys.path` to allow importing modules from the parent directory.

## Workflow Instructions
*   **Feature Implementation:**
    *   **New Features:** Always create a new branch.
        *   Naming convention: `feature/<feature-name>`
        *   Command: `git checkout -b feature/<feature-name>`
    *   **Existing Features:** Continue working on the same branch.
        *   Command: `git checkout feature/<feature-name>`
        *   Update: `git pull origin feature/<feature-name>` (if necessary)