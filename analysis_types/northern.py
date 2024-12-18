import os
import cv2
import pandas as pd
from lane_detector import LaneDetector
from lane_extractor import LaneExtractor
from band_detector import BandDetector


class NorthernBlotAnalysis:
    """
    Handles analysis logic for Northern blots, including lane detection,
    extraction, and band detection with parameters optimized for RNA detection.
    """

    def __init__(self, min_area_factor: float = 0.001, max_area_factor: float = 0.15, intensity_threshold: int = 40):
        """
        Initialize parameters specific to Northern blot analysis.

        Args:
            min_area_factor (float): Minimum band area factor as a fraction of lane area.
            max_area_factor (float): Maximum band area factor as a fraction of lane area.
            intensity_threshold (int): Minimum intensity threshold for valid bands.
        """
        self.min_area_factor = min_area_factor
        self.max_area_factor = max_area_factor
        self.intensity_threshold = intensity_threshold

    def analyze(self, image_path: str, output_dir: str):
        """
        Perform analysis on a Northern blot image.

        Args:
            image_path (str): Path to the input image.
            output_dir (str): Directory to save output results.
        """
        print("Running Northern Blot analysis...")

        # Prepare output directories
        lanes_dir = os.path.join(output_dir, "lanes")
        annotated_dir = os.path.join(output_dir, "annotated_lanes")
        os.makedirs(lanes_dir, exist_ok=True)
        os.makedirs(annotated_dir, exist_ok=True)

        # Load the gel image
        gel_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Step 1: Detect lanes
        print("Detecting lanes...")
        lane_detector = LaneDetector(height_factor=0.4, distance=40)
        lane_ranges = lane_detector.detect_lanes(
            gel_image, smooth_sigma=2.0, visualize=True)

        # Step 2: Extract lanes
        print("Extracting lanes...")
        lane_extractor = LaneExtractor(padding_factor=0.1)
        lanes_metadata = lane_extractor.extract_lanes(
            gel_image, lane_ranges, output_dir=lanes_dir)

        # Step 3: Detect and annotate bands
        print("Detecting bands...")
        band_detector = BandDetector(
            min_area_factor=self.min_area_factor,
            max_area_factor=self.max_area_factor,
            intensity_threshold=self.intensity_threshold
        )
        all_band_metadata = []

        for lane in lanes_metadata:
            # Detect bands in the lane
            lane_image = lane["image"]
            bands = band_detector.detect_bands(lane_image)

            # Merge overlapping or nearby bands
            merged_bands = band_detector.merge_bands(bands, merge_distance=5)

            # Annotate the lane with detected bands
            annotated_lane = band_detector.annotate_bands(
                lane_image, merged_bands)

            # Save annotated lane
            lane_index = lane["index"]
            annotated_path = os.path.join(
                annotated_dir, f"annotated_lane_{lane_index}.png")
            cv2.imwrite(annotated_path, annotated_lane)
            print(f"Annotated Lane {lane_index} saved to {annotated_path}")

            # Append metadata for each band
            for band in merged_bands:
                band["lane_index"] = lane_index
                all_band_metadata.append(band)

        # Step 4: Export band metadata to CSV
        print("Exporting band metadata...")
        metadata_path = os.path.join(output_dir, "band_metadata.csv")
        pd.DataFrame(all_band_metadata).to_csv(metadata_path, index=False)
        print(f"Band metadata saved to {metadata_path}")
