import cv2
import numpy as np
    
def lowest_start(phases, imposed_order):
    """
    Determines the earliest phase (according to imposed order) that occurs with significant frequency in the given list of phases.

    Parameters:
    phases (list): A list of phases to analyze.
    imposed_order (list): The logical sequence of phases.

    Returns:
    str: The earliest phase from the imposed order that appears frequently enough in the input list.
    """
    
    # Create a dictionary to map each phase to its index in the imposed order
    phase_idx_dict = {phase: idx for idx, phase in enumerate(imposed_order)}
    
    lowest_idx = 100  # Initialize to a large number to find the minimum index
    for phase in phases:
        # Check if the current phase's index is lower than the current lowest index
        # and if the phase appears in at least 20% of the input list
        if phase_idx_dict[phase] < lowest_idx and (phases.count(phase)/len(phases)) >= 0.1:
            lowest_idx = phase_idx_dict[phase]  # Update the lowest index
    return imposed_order[lowest_idx]  # Return the phase corresponding to the lowest index


def accumulator_post_processing(predictions, thresholds, imposed_order, diverge_thresh, lowest_thresh=50):
    """
    Smooths the phase predictions of a surgical procedure to enforce a logical and ordered progression.
    
    Parameters:
    predictions (list): The list of phase predictions to be smoothed.
    thresholds (dict): A dictionary mapping each phase to its threshold, which is the number of consecutive 
                       frames required to confirm a phase transition.
    imposed_order (list): The logical order of phases to be enforced.
    
    Returns:
    list: The smoothed list of phase predictions.
    """
    smoothed_predictions = []  # List to store the final smoothed predictions
    current_phase = lowest_start(predictions[:lowest_thresh], imposed_order) # Initial phase based on the first 50 predictions
    phase_idx = imposed_order.index(current_phase)  # Index of the current phase in the imposed order
    idx = 0  # Index for iterating through predictions
    diverging_count = 0  # Counter for tracking divergence from the current phase
    divergent_phase_idx = None  # Initialize divergent_phase_idx
    divergent_phase = None  # Initialize divergent_phase
    diverging_idx = None  # Initialize diverging_idx
    
    while idx < len(predictions):
        # Handle prolonged divergence from the current phase by introducing next phase at divergence point
        if diverging_count > diverge_thresh and divergent_phase_idx != len(imposed_order) - 1:
            idx = diverging_idx + 1  # Move index to the next prediction after divergence
            predictions[idx] = imposed_order[divergent_phase_idx + 1] # Set next phase in the imposed order
            diverging_count = 0  # Reset divergence counter
            smoothed_predictions = smoothed_predictions[:idx]  # Trim smoothed predictions up to current index
            current_phase = imposed_order[divergent_phase_idx + 1]  # Update current phase
            phase_idx = divergent_phase_idx + 1  # Update phase index
        prediction = predictions[idx]  # Current prediction
        prediction_idx = imposed_order.index(prediction)  # Index of the current prediction in imposed order
        
        # Check if the prediction suggests a transition to the next phase
        if prediction != current_phase and prediction_idx == phase_idx + 1:
            candidates = []  # List to collect potential candidates for phase transition
            candidate_phase = imposed_order[phase_idx + 1]  # Next phase in imposed order
            threshold = thresholds[prediction]  # Threshold for the candidate phase
            
            # Ensure threshold does not exceed remaining predictions
            threshold = threshold if idx + threshold < len(predictions) else len(predictions) - idx - 1
            
            # Collect predictions for the candidate phase
            for pred in predictions[idx:idx + threshold]:
                if pred == candidate_phase: candidates.append(pred)
                else: break
                    
            # Confirm phase transition if candidates meet the threshold
            if len(candidates) == threshold:
                smoothed_predictions.extend(candidates)
                idx += threshold  # Move index past the confirmed candidates
                current_phase = candidate_phase  # Update current phase
                phase_idx += 1  # Update phase index
            else:
                # If candidates do not meet threshold, retain current phase
                unchanged_phases = [current_phase]*len(candidates)
                smoothed_predictions.extend(unchanged_phases)
                idx += len(candidates)  # Move index past the evaluated candidates

        elif prediction != current_phase:
            # Track divergence from the current phase
            divergent_phase = current_phase
            divergent_phase_idx = phase_idx
            diverging_idx = idx
            diverging_count += 1
            smoothed_predictions.append(current_phase)  # Retain current phase
            idx += 1  # Move to the next prediction
        else:
            # Reset divergence counter and retain current prediction
            diverging_count = 0
            smoothed_predictions.append(prediction)
            idx += 1  # Move to the next prediction
    
    # Ensure the first smoothed prediction is not a single outlier
    if smoothed_predictions.count(smoothed_predictions[0]) == 1:
        smoothed_predictions[0] = smoothed_predictions[1]
    return smoothed_predictions


def adjust_values(strings, threshold, percentage):
    """
    Adjusts the values in the string list based on the specified percentage and threshold,
    without modifying the original list. The percentage is calculated based on the total
    of before and after values.

    :param strings: List of strings.
    :param threshold: Number of values before and after to consider.
    :param percentage: Percentage threshold to determine if a value should be changed.
    :return: Modified list of strings.
    """
    n = len(strings)
    adjusted_strings = strings.copy()

    # Iterate through the list to identify and adjust values
    for i in range(n):
        before = strings[max(0, i - threshold):i]
        after = strings[i:min(n, i + threshold + 1)]
        surrounding = before + after

        # Count the occurrences of each value in the surrounding values
        surrounding_count = {val: surrounding.count(val) for val in set(surrounding)}

        # Determine the percentage for the most common value in surrounding values
        max_count = max(surrounding_count.values(), default=0)
        total_surrounding = len(surrounding)

        common_percentage = (max_count / total_surrounding * 100) if total_surrounding else 0
        current_cls = adjusted_strings[i]
        current_percentage = (surrounding_count[current_cls]/ total_surrounding * 100) if total_surrounding else 100

        # If the current value is different from the most common value and the percentage is met, adjust it
        most_common_value = max(surrounding_count, key=surrounding_count.get, default=None)

        if adjusted_strings[i] != most_common_value and (common_percentage >= percentage or current_percentage <= 20):
            adjusted_strings[i] = most_common_value

    return adjusted_strings


def convert_seconds_to_timestamp(seconds_input):
    """
    Convert a duration given in seconds into a timestamp format (HH:MM:SS).

    Args:
    - seconds (float): Duration in seconds.

    Returns:
    - str: Timestamp formatted as 'HH:MM:SS'.
    """
    hours, remainder = divmod(seconds_input, 3600)
    minutes, seconds = divmod(remainder, 60)
    frac = str(seconds_input).split('.')[1][:2]
    hours_int = int(hours)
    minutes_int = int(minutes)
    seconds_int = int(seconds)
    frac_int = int(frac)

    formatted_time = f"{hours_int:02d}:{minutes_int:02d}:{seconds_int:02d}.{frac_int:02d}"
    return formatted_time


def is_pixelated(frame, variance_threshold=30, block_diff_threshold=100):
    # check the image 
    if frame is None:
        print(f"Failed to load image.")
        return False

    # Center crop the image
    h, w, _ = frame.shape
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped_image = frame[start_y:start_y + min_dim, start_x:start_x + min_dim]
    
    # Convert to grayscale
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the variance of the Laplacian
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if laplacian_var < variance_threshold:
        return True  # The image is likely blurry or pixelated
    
    # Check blockiness by resizing
    small = cv2.resize(cropped_image, (16, 16), interpolation=cv2.INTER_LINEAR)
    resized_back = cv2.resize(small, (cropped_image.shape[1], cropped_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Calculate the difference between the original and resized-back image
    diff = cv2.absdiff(cropped_image, resized_back)
    mean_diff = np.mean(diff)
    
    if mean_diff > block_diff_threshold:
        return True  # The image is likely pixelated
    
    return False

def calculate_red_percentage(image):
    if image is None:
        raise ValueError("Could not read the image.")

    # Convert to HSV (Hue, Saturation, Value) color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range of red color in HSV
    # Adjust these ranges according to your requirement for red color
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create masks to detect red
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Calculate percentage of red
    red_pixels = np.sum(red_mask == 255)
    total_pixels = image.shape[0] * image.shape[1]
    red_percentage = (red_pixels / total_pixels) * 100

    return red_percentage

