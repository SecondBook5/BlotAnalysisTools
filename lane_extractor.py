# Import necessary libraries
import cv2  # For image processing
import numpy as np  # For numerical operations
import os  # For file handling
from concurrent.futures import ThreadPoolExecutor  # For parallel processing
import logging  # For controlled logging of messages


class LaneExtractor:
    """
    A class to extract individual lanes from a gel image using lane boundaries.
    Features:
        - Dynamic padding based on lane width
        - Intensity normalization for consistent contrast
        - Parallel processing for speed
        - Optional saving of extracted lanes
    """

    def __init__(self, padding_factor: float = 0.1):
        """
        Initialize the LaneExtractor with dynamic padding as a fraction of lane width.

        Args:
            padding_factor (float): Fraction of lane width to use as padding.
        """
        self.padding_factor = padding_factor  # Store padding factor for dynamic padding
        # Configure the logging system to display messages with level INFO or higher
        logging.basicConfig(level=logging.INFO,
                            format="%(levelname)s: %(message)s")

    def _normalize_lane(self, lane: np.ndarray) -> np.ndarray:
        """
        Normalize the intensity of a single lane to improve contrast.

        Args:
            lane (np.ndarray): The cropped lane image.

        Returns:
            np.ndarray: Normalized lane image with intensities between 0 and 255.
        """
        # Normalize image intensities using min-max normalization
        return cv2.normalize(lane, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    def extract_lanes(self, image: np.ndarray, lane_ranges: list, output_dir: str = None) -> list:
        """
        Extract lanes as sub-images based on provided lane boundaries and optionally save them.

        Args:
            image (np.ndarray): Original grayscale gel image.
            lane_ranges (list): List of lane boundaries as (left, right) tuples.
            output_dir (str): Directory to save the extracted lanes. If None, lanes are not saved.

        Returns:
            list: List of dictionaries with lane metadata (index, left, right, and lane image).
        """

        def process_lane(i, left, right):
            """
            Inner function to process each lane by cropping, normalizing, and saving it.

            Args:
                i (int): Index of the lane.
                left (int): Left boundary of the lane.
                right (int): Right boundary of the lane.

            Returns:
                dict: Metadata of the processed lane.
            """
            # Step 1: Calculate dynamic padding based on the lane width
            lane_width = right - left  # Compute the width of the lane
            # Padding is a fraction of the lane width
            padding = int(self.padding_factor * lane_width)

            # Step 2: Apply padding while ensuring boundaries are within image limits
            # Ensure left boundary doesn't go negative
            left_padded = max(left - padding, 0)
            # Ensure right boundary is within the image width
            right_padded = min(right + padding, image.shape[1])

            # Step 3: Crop the lane from the image
            lane = image[:, left_padded:right_padded]

            # Step 4: Normalize the lane intensity for consistent contrast
            lane = self._normalize_lane(lane)

            # Step 5: Save the lane to the output directory if specified
            if output_dir:
                # Ensure the output directory exists
                os.makedirs(output_dir, exist_ok=True)
                # Define the file path
                lane_path = os.path.join(output_dir, f"lane_{i+1}.png")
                cv2.imwrite(lane_path, lane)  # Save the lane as a PNG image
                # Log the save operation
                logging.info(f"Saved Lane {i+1} to {lane_path}")

            # Step 6: Return metadata about the processed lane
            return {"index": i + 1, "left": left_padded, "right": right_padded, "image": lane}

        # Step 7: Process lanes in parallel for efficiency
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                lambda args: process_lane(*args), enumerate(lane_ranges)))

        # Step 8: Return the list of lane metadata
        return results
