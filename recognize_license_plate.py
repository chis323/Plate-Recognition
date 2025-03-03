import os
import cv2
from datetime import datetime
from ultralytics import YOLO  # YOLOv8
import pytesseract  # Tesseract OCR

# Set the path to the Tesseract executable (update this path based on your installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Source path containing all images
SOURCE_PATH = os.path.dirname(os.path.abspath(__file__))  # Get the folder of the script dynamically

# Create an output folder if it doesn't exist
OUTPUT_FOLDER = os.path.join(SOURCE_PATH, "output")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def preprocess_plate_image(plate_region):
    """Preprocess the cropped license plate image for better OCR accuracy."""
    # Convert to grayscale
    gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Apply morphological operations to clean up the image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Resize the image for better readability
    resized = cv2.resize(morph, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    return resized

def correct_ocr_misclassifications(text):
    """Correct common OCR misclassifications."""
    # Replace '8' with 'B' if it makes sense in the context of a license plate
    corrected_text = text.replace('8', 'B')
    return corrected_text

def recognize_license_plate(img_path):
    """Recognize a license plate in an image using YOLOv8 and save processed images in the output folder."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Error: Image file '{img_path}' not found. Please check the path.")

    start_time = datetime.now()

    # Read image with OpenCV
    print(f"Loading image from: {img_path}")
    img = cv2.imread(img_path)
    
    if img is None:
        raise ValueError(f"Error: Unable to read the image '{img_path}'. Please check the file format and integrity.")
    print("Image loaded successfully.")

    # Get image size
    height, width = img.shape[:2]

    # Scale image for better recognition (optional)
    img = cv2.resize(img, (800, int((height * 800) / width)))

    # Save the resized image in the output folder
    resized_image_path = os.path.join(OUTPUT_FOLDER, "resized_input.jpg")
    cv2.imwrite(resized_image_path, img)

    # Load the custom-trained YOLOv8 model
    model = YOLO(r"C:\Users\Chis Bogdan\Desktop\NPR\recognize-license-plate-master\runs\detect\license_plate_detection\weights\best.pt")  # Use raw string

    # Perform object detection
    results = model(img)
    print(f"Detection results: {results}")

    # Extract detected objects
    detected_plate = None
    plate_number = None
    for result in results:
        boxes = result.boxes  # Bounding boxes
        for box in boxes:
            class_id = box.cls  # Class ID (if multiple classes are detected)
            confidence = box.conf  # Confidence score
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

            # Assuming class_id 0 is for license plates (customize based on your model)
            if class_id == 0 and confidence > 0.3:  # Lower the confidence threshold
                detected_plate = "Detected License Plate"

                # Crop the license plate region
                plate_region = img[y1:y2, x1:x2]

                # Preprocess the cropped license plate image
                processed_plate = preprocess_plate_image(plate_region)

                # Use Tesseract OCR to extract text from the license plate region
                plate_number = pytesseract.image_to_string(processed_plate, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                print(f"Raw OCR output: {plate_number}")
                plate_number = plate_number.strip()  # Remove any extra whitespace

                # Clean the extracted text (remove unwanted characters)
                plate_number = ''.join(e for e in plate_number if e.isalnum())

                # Correct common OCR misclassifications
                plate_number = correct_ocr_misclassifications(plate_number)

                # Draw the license plate text on the image
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Draw bounding box
                cv2.putText(img, f"License Plate: {plate_number}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if detected_plate and plate_number:
        print(f"Detected License Plate: {plate_number}")

        # Save the detected license plate image
        output_detected_image_path = os.path.join(OUTPUT_FOLDER, "detected_license_plate.jpg")
        cv2.imwrite(output_detected_image_path, img)

        print(f'Total time: {datetime.now() - start_time}')
        
        # Save and display the final output image
        output_image_path = os.path.join(OUTPUT_FOLDER, "final_output.jpg")
        cv2.imwrite(output_image_path, img)
        print(f"Output image saved to: {output_image_path}")

        # Display the detected license plate image
        cv2.imshow('Detected License Plate', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    raise ValueError(f"Error: No license plate detected in the image '{img_path}'.")

# Main execution
if __name__ == "__main__":
    print('---------- Start Recognizing License Plate --------')

    # Set the image path
    image_filename = os.path.join(SOURCE_PATH, '1.jpg')

    try:
        recognize_license_plate(image_filename)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")

    print('---------- End ----------')