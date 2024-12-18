import cv2  # For image processing
import numpy as np  # For numerical operations


class BandDetector:
    """
    A class to detect and analyze bands within a lane image, with support for:
        - Adaptive thresholding
        - Intensity-based filtering
        - Band merging for overlapping detections
        - Comprehensive metadata generation
    """

    def __init__(self, min_area_factor: float = 0.001, max_area_factor: float = 0.1, intensity_threshold: int = 50):
        """
        Initialize BandDetector with dynamic filtering parameters.

        Args:
            min_area_factor (float): Minimum area as a fraction of lane area for valid bands.
            max_area_factor (float): Maximum area as a fraction of lane area for valid bands.
            intensity_threshold (int): Minimum mean intensity of a band to be considered valid.
        """
        self.min_area_factor = min_area_factor  # Minimum area threshold factor
        self.max_area_factor = max_area_factor  # Maximum area threshold factor
        self.intensity_threshold = intensity_threshold  # Minimum mean intensity threshold

    def detect_bands(self, lane: np.ndarray) -> list:
        """
        Detect bands within a lane image using adaptive thresholding and contour detection.

        Args:
            lane (np.ndarray): Cropped grayscale lane image.

        Returns:
            list: List of band metadata, including bounding boxes, area, and intensity.
        """
        # Step 1: Adaptive thresholding to handle varying intensity
        binary_image = cv2.adaptiveThreshold(
            lane, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Step 2: Calculate dynamic area thresholds based on lane size
        lane_height, lane_width = lane.shape[:2]
        min_area = int(self.min_area_factor * lane_height * lane_width)
        max_area = int(self.max_area_factor * lane_height * lane_width)

        # Step 3: Detect contours in the binary image
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 4: Process contours to filter and collect metadata
        bands = []
        for cnt in contours:
            # Calculate the bounding box and area of the contour
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)

            # Skip small or excessively large contours
            if not (min_area <= area <= max_area):
                continue

            # Create a mask for the contour to calculate intensity
            mask = np.zeros_like(binary_image)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_intensity = cv2.mean(lane, mask=mask)[0]

            # Skip bands with insufficient intensity
            if mean_intensity < self.intensity_threshold:
                continue

            # Append band metadata
            bands.append({
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "area": area,
                "intensity": mean_intensity
            })

        print(f"Detected {len(bands)} valid bands.")
        return bands

    def merge_bands(self, bands: list, merge_distance: int = 5) -> list:
        """
        Merge nearby bands based on proximity to handle overlapping detections.

        Args:
            bands (list): List of band bounding boxes and metadata.
            merge_distance (int): Maximum distance between bands to consider merging.

        Returns:
            list: List of merged bands with updated metadata.
        """
        # Sort bands by their x-coordinate for efficient merging
        bands.sort(key=lambda b: b["x"])

        merged_bands = []
        for band in bands:
            if merged_bands and abs(band["x"] - (merged_bands[-1]["x"] + merged_bands[-1]["width"])) < merge_distance:
                # Merge with the previous band
                prev_band = merged_bands.pop()
                merged_x = min(band["x"], prev_band["x"])
                merged_y = min(band["y"], prev_band["y"])
                merged_width = max(
                    band["x"] + band["width"], prev_band["x"] + prev_band["width"]) - merged_x
                merged_height = max(
                    band["y"] + band["height"], prev_band["y"] + prev_band["height"]) - merged_y
                merged_area = prev_band["area"] + band["area"]
                merged_intensity = (
                    prev_band["intensity"] + band["intensity"]) / 2
                merged_bands.append({
                    "x": merged_x,
                    "y": merged_y,
                    "width": merged_width,
                    "height": merged_height,
                    "area": merged_area,
                    "intensity": merged_intensity
                })
            else:
                merged_bands.append(band)

        print(f"Merged to {len(merged_bands)} bands.")
        return merged_bands

    def annotate_bands(self, lane: np.ndarray, bands: list) -> np.ndarray:
        """
        Annotate detected bands on the lane image.

        Args:
            lane (np.ndarray): Cropped grayscale lane image.
            bands (list): List of band bounding boxes and metadata.

        Returns:
            np.ndarray: Annotated lane image with bands highlighted.
        """
        # Convert grayscale lane to BGR for colored annotation
        annotated_lane = cv2.cvtColor(lane, cv2.COLOR_GRAY2BGR)

        # Draw rectangles and metadata for each detected band
        for band in bands:
            x, y, w, h = band["x"], band["y"], band["width"], band["height"]
            cv2.rectangle(annotated_lane, (x, y), (x + w, y + h),
                          (0, 255, 0), 1)  # Green rectangle
            cv2.putText(
                annotated_lane, f"Int:{int(band['intensity'])}",
                (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
            )  # Intensity text

        return annotated_lane
