import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
import os
import urllib.request

def download_model(model_path):
    if not os.path.exists(model_path):
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        print(f"Downloading model to {model_path}...")
        try:
            urllib.request.urlretrieve(url, model_path)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download model: {e}")
            return False
    return True

def slim_face_effect(image_path):
    # 1. Face Detection and Facial Landmark Extraction
    
    # Ensure model exists
    model_path = os.path.abspath("face_landmarker.task")
    if not download_model(model_path):
        print("Error: Model file could not be downloaded.")
        return

    # Create an FaceLandmarker object.
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=False,
                                           output_facial_transformation_matrixes=False,
                                           num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load the input image from a numpy array.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    # Detect face landmarks from the input image.
    detection_result = detector.detect(mp_image)

    if not detection_result.face_landmarks:
        print("No face detected.")
        return

    landmarks = detection_result.face_landmarks[0]
    h, w, _ = image.shape

    # Convert landmarks to numpy array
    points = np.array([(int(l.x * w), int(l.y * h)) for l in landmarks])

    # 2. Selecting Target Landmarks for Deformation (Jawline)
    # MediaPipe Face Mesh indices for jawline (approximate)
    # Left side (from chin up): 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    # Right side (from chin up): 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338
    
    # Simplified selection for demonstration (Chin to Ear)
    # Left Jaw: 
    left_jaw_indices = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148]
    # Right Jaw: 
    right_jaw_indices = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377]
    
    # Chin point
    chin_index = 152
    
    # Combine indices
    jaw_indices = left_jaw_indices + [chin_index] + right_jaw_indices
    
    src_points = points[jaw_indices]
    
    # 3. Computing the New Jawline for the Slim-Face Effect
    dst_points = src_points.copy()
    
    # Calculate center of the face (approximate using nose tip)
    nose_tip_index = 1
    nose_tip = points[nose_tip_index]
    
    # Apply slimming logic
    # "Landmarks near the upper face (smaller y) use ratios close to 1, 
    # while those near the lower face (larger y) use smaller ratios."
    
    # Find min and max y of the jawline to normalize
    min_y = np.min(src_points[:, 1])
    max_y = np.max(src_points[:, 1])
    
    for i, point in enumerate(src_points):
        # Calculate vertical position ratio (0 at top of jaw, 1 at chin)
        # Actually, for slimming, we want more effect at the bottom (chin area)
        # But usually jawline points are lower than cheeks.
        
        # Let's define a simple scaling towards the center x-axis
        # The scaling factor depends on y.
        
        # Normalized y position relative to the jawline height
        # 0.0 = highest point of jaw (near ear), 1.0 = lowest point (chin)
        y_ratio = (point[1] - min_y) / (max_y - min_y + 1e-6)
        
        # Slimming strength: stronger at the bottom
        # strength = 0.0 (no change) to 0.2 (20% slimmer)
        strength = 0.15 * y_ratio 
        
        # Direction vector from point to center (horizontal only for simple slimming)
        # We want to pull points towards the center line (x = nose_tip[0])
        direction = nose_tip[0] - point[0]
        
        # Apply shift
        dst_points[i, 0] = int(point[0] + direction * strength)

    # Add image corners as control points to prevent global distortion
    corners = np.array([
        [0, 0],
        [w - 1, 0],
        [0, h - 1],
        [w - 1, h - 1]
    ])
    
    # Add some extra points around the face to keep the rest of the face stable?
    # For this assignment, "add four corners as the control points as well"
    
    all_src_points = np.vstack([src_points, corners])
    all_dst_points = np.vstack([dst_points, corners])
    
    # 4. Constructing the Deformation Field Using RBFInterpolator (Thin-Plate Spline)
    # We need to map destination pixels back to source pixels (inverse mapping) for cv2.remap
    # So we fit: f(dst) -> src
    
    # RBFInterpolator expects (N, D) arrays
    # kernel='thin_plate_spline'
    rbf = RBFInterpolator(all_dst_points, all_src_points, kernel='thin_plate_spline')
    
    # Create a grid of coordinates for the entire image
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    
    # Predict source coordinates for each grid point
    # This can be memory intensive for large images. 
    # For efficiency, one might warp a coarser grid and interpolate, but here we do full resolution as per typical assignment scale.
    
    # To avoid OOM on large images, we can process in chunks, but let's try direct first.
    print("Computing deformation field... this might take a moment.")
    
    # Splitting into chunks to be safe with memory
    chunk_size = 10000
    map_x = np.zeros(h * w, dtype=np.float32)
    map_y = np.zeros(h * w, dtype=np.float32)
    
    for i in range(0, len(grid_points), chunk_size):
        chunk = grid_points[i:i+chunk_size]
        predicted = rbf(chunk)
        map_x[i:i+chunk_size] = predicted[:, 0]
        map_y[i:i+chunk_size] = predicted[:, 1]
        
    map_x = map_x.reshape(h, w)
    map_y = map_y.reshape(h, w)
    
    # Remap
    output_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    # Original with landmarks
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.scatter(src_points[:, 0], src_points[:, 1], c='red', s=10, label='Original Jawline')
    ax1.scatter(dst_points[:, 0], dst_points[:, 1], c='blue', s=10, label='Target Jawline')
    ax1.set_title("Original Image with Control Points")
    ax1.legend()
    ax1.axis('off')
    # Add watermark to the first image
    ax1.text(0.99, 0.01, '© 2025 Huang Yu Chien. All rights reserved.', 
             transform=ax1.transAxes, ha='right', va='bottom', fontsize=10, color='white', alpha=0.7)
    
    # Result
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    ax2.set_title("Slim-Face Effect Result")
    ax2.axis('off')
    # Add watermark to the second image
    ax2.text(0.99, 0.01, '© 2025 Huang Yu Chien. All rights reserved.', 
             transform=ax2.transAxes, ha='right', va='bottom', fontsize=10, color='white', alpha=0.7)
    
    plt.tight_layout()
    
    # Add watermark to the output image (for saving)
    h_out, w_out = output_image.shape[:2]
    text = "© 2025 Huang Yu Chien. All rights reserved."
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (255, 255, 255) # White
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Position: Bottom right
    x = w_out - text_width - 10
    y = h_out - 10
    
    cv2.putText(output_image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    # Save result first
    dir_name, file_name = os.path.split(image_path)
    output_filename = "slim_" + file_name
    output_path = os.path.join(dir_name, output_filename)

    cv2.imwrite(output_path, output_image)
    print(f"Result saved to {output_path}")

    plt.show()

if __name__ == "__main__":
    import os
    
    # Get image path from user input
    image_path = input("Please enter the path to the image file (default: face.jpg): ").strip()
    
    if not image_path:
        image_path = "face.jpg"

    if os.path.exists(image_path):
        slim_face_effect(image_path)
    else:
        print(f"File '{image_path}' not found.")
        if image_path == "face.jpg":
            print("Creating a dummy 'face.jpg' for demonstration...")
            dummy_img = np.ones((400, 400, 3), dtype=np.uint8) * 255
            # Draw a face
            cv2.circle(dummy_img, (200, 200), 100, (200, 200, 255), -1) # Face
            cv2.circle(dummy_img, (170, 180), 10, (0, 0, 0), -1) # Left Eye
            cv2.circle(dummy_img, (230, 180), 10, (0, 0, 0), -1) # Right Eye
            cv2.ellipse(dummy_img, (200, 240), (30, 10), 0, 0, 180, (0, 0, 0), 2) # Mouth
            cv2.imwrite("face.jpg", dummy_img)
            slim_face_effect("face.jpg")