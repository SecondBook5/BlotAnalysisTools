import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt


class LaneDetector:
    """
    A class to detect lanes in a gel image using vertical intensity projection.

    Methods:
        detect_lanes(image: np.ndarray, smooth_sigma: float, visualize: bool) -> list:
            Detect lanes and return their ranges as (left, right) boundaries.
    """

    def __init__(self, height_factor: float = 0.5, distance: int = 30):
        """
        Initialize LaneDetector with peak detection parameters.

        Args:
            height_factor (float): Factor of the max intensity for peak detection threshold.
            distance (int): Minimum distance between adjacent peaks.
        """
        self.height_factor = height_factor
        self.distance = distance

    def detect_lanes(self, image: np.ndarray, smooth_sigma: float = 2.0, visualize: bool = True) -> list:
        """
        Detect lanes in a thresholded gel image.

        Args:
            image (np.ndarray): Grayscale or thresholded gel image.
            smooth_sigma (float): Sigma for Gaussian smoothing of the vertical projection.
            visualize (bool): Whether to visualize the projection and peaks.

        Returns:
            list: List of lane boundaries as (left, right) tuples.
        """
        # Step 1: Calculate the vertical intensity projection
        vertical_projection = np.sum(image, axis=0)

        # Step 2: Smooth the projection to reduce noise
        smoothed_projection = gaussian_filter1d(
            vertical_projection, sigma=smooth_sigma)

        # Step 3: Determine dynamic height threshold
        max_intensity = np.max(smoothed_projection)
        height_threshold = self.height_factor * max_intensity

        # Step 4: Detect peaks in the smoothed projection
        peaks, _ = find_peaks(smoothed_projection,
                              height=height_threshold, distance=self.distance)

        # Step 5: Convert peaks to lane boundaries (left, right edges)
        lane_width = self._estimate_lane_width(peaks)
        lane_ranges = [(max(peak - lane_width // 2, 0), min(peak + lane_width // 2, image.shape[1]))
                       for peak in peaks]

        # Step 6: Visualization (optional)
        if visualize:
            self._visualize_lanes(smoothed_projection, peaks, lane_ranges)

        print(f"Detected lanes at ranges: {lane_ranges}")
        return lane_ranges

    @staticmethod
    def _estimate_lane_width(peaks: list) -> int:
        """
        Estimate lane width based on average spacing between detected peaks.

        Args:
            peaks (list): List of peak positions.

        Returns:
            int: Estimated lane width.
        """
        if len(peaks) < 2:
            return 40  # Default lane width if only one peak
        distances = np.diff(peaks)
        return int(np.median(distances))

    @staticmethod
    def _visualize_lanes(projection: np.ndarray, peaks: list, ranges: list):
        """
        Visualize the vertical intensity projection and detected lanes.

        Args:
            projection (np.ndarray): Smoothed vertical projection.
            peaks (list): Detected peaks.
            ranges (list): Lane boundaries as (left, right).
        """
        plt.figure(figsize=(10, 4))
        plt.plot(projection, label="Smoothed Vertical Projection")
        plt.scatter(peaks, projection[peaks],
                    color='red', label="Detected Peaks")
        for left, right in ranges:
            plt.axvspan(left, right, color='green', alpha=0.3,
                        label="Lane Range" if left == ranges[0][0] else "")
        plt.title("Vertical Intensity Projection with Detected Lanes")
        plt.xlabel("Column Index")
        plt.ylabel("Summed Intensity")
        plt.legend()
        plt.show()
