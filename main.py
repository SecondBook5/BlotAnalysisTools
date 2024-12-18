import argparse
import os
import cv2
import pandas as pd
from lane_detector import LaneDetector
from lane_extractor import LaneExtractor
from band_detector import BandDetector
from analysis_types.western import WesternBlotAnalysis
from analysis_types.northern import NorthernBlotAnalysis
from analysis_types.footprinting import DNaseFootprintingAnalysis


def run_generic_pipeline(image_path: str, output_dir: str, min_area_factor: float = 0.001, max_area_factor: float = 0.1, intensity_threshold: int = 50):
    """
    Generic pipeline to process a gel image for lane and band detection.

    Args:
        image_path (str): Path to the gel image.
        output_dir (str): Directory to save output results.
        min_area_factor (float): Minimum band area factor as a fraction of lane area.
        max_area_factor (float): Maximum band area factor as a fraction of lane area.
        intensity_threshold (int): Minimum intensity threshold for valid bands.
    """
    # Step 1: Prepare output directories
    lanes_dir = os.path.join(output_dir, "lanes")
    annotated_dir = os.path.join(output_dir, "annotated_lanes")
    os.makedirs(lanes_dir, exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)

    # Step 2: Load the gel image
    gel_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Step 3: Detect lanes
    print("Detecting lanes...")
    lane_detector = LaneDetector(height_factor=0.4, distance=40)
    lane_ranges = lane_detector.detect_lanes(
        gel_image, smooth_sigma=2.0, visualize=True)

    # Step 4: Extract lanes
    print("Extracting lanes...")
    lane_extractor = LaneExtractor(padding_factor=0.1)
    lanes_metadata = lane_extractor.extract_lanes(
        gel_image, lane_ranges, output_dir=lanes_dir)

    # Step 5: Detect and annotate bands
    print("Detecting bands...")
    band_detector = BandDetector(
        min_area_factor=min_area_factor,
        max_area_factor=max_area_factor,
        intensity_threshold=intensity_threshold
    )
    all_band_metadata = []

    for lane in lanes_metadata:
        # Detect bands in the lane
        lane_image = lane["image"]
        bands = band_detector.detect_bands(lane_image)

        # Merge nearby bands
        merged_bands = band_detector.merge_bands(bands, merge_distance=5)

        # Annotate the lane with detected bands
        annotated_lane = band_detector.annotate_bands(lane_image, merged_bands)

        # Save annotated lane
        lane_index = lane["index"]
        annotated_path = os.path.join(
            annotated_dir, f"annotated_lane_{lane_index}.png")
        cv2.imwrite(annotated_path, annotated_lane)
        print(f"Annotated Lane {lane_index} saved to {annotated_path}")

        # Append band metadata
        for band in merged_bands:
            band["lane_index"] = lane_index
            all_band_metadata.append(band)

    # Step 6: Export band metadata to CSV
    print("Exporting band metadata...")
    metadata_path = os.path.join(output_dir, "band_metadata.csv")
    band_metadata_df = pd.DataFrame(all_band_metadata)
    band_metadata_df.to_csv(metadata_path, index=False)
    print(f"Band metadata saved to {metadata_path}")

    print("Pipeline completed successfully!")


def main():
    """
    Command-line interface for BlotAnalysisTools. Supports modular analysis types and generic pipeline execution.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="BlotAnalysisTools: Analyze various gel and blot images.")
    parser.add_argument(
        "--type",
        choices=["western", "northern", "footprinting", "generic"],
        required=True,
        help="Type of analysis to perform (e.g., western, northern, footprinting, generic).",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input gel or blot image.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory to save output results.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Whether to visualize lane detection during the analysis.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Validate input
    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        return

    # Validate output directory
    os.makedirs(args.output, exist_ok=True)

    # Dispatch to the appropriate analysis type
    if args.type == "western":
        analysis = WesternBlotAnalysis()
        analysis.analyze(args.input, args.output)
    elif args.type == "northern":
        analysis = NorthernBlotAnalysis()
        analysis.analyze(args.input, args.output)
    elif args.type == "footprinting":
        analysis = DNaseFootprintingAnalysis()
        analysis.analyze(args.input, args.output)
    elif args.type == "generic":
        # Run the generic pipeline with default parameters
        run_generic_pipeline(args.input, args.output)
    else:
        print(f"Error: Unsupported analysis type '{args.type}'.")


if __name__ == "__main__":
    main()
