import cv2
import pytesseract
import dotenv
import os
import google.generativeai as genai

dotenv.load_dotenv()

GEMINI_API_KEY = os.getenv
genai.configure(api_key=GEMINI_API_KEY)

# initializing the gemini model
model = genai.GenerativeModel('gemini-pro')

def preprocess_image(image_path):
    # load the image
    image = cv2.imread(image_path)

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply thresholding
    _, thresholded = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    return thresholded

def extract_text(image):
    # using pytesseract to extract text
    text = pytesseract.image_to_string(image)
    return text

def ai_extract(text):
    # defining a prompt for Gemini
    prompt = f"""
    Extract the following details from the receipt text below and return them as a JSON object:
    - Total Amount
    - Date
    - Items Purchased (as a list)
    - Payment Method

    Receipt Text:
    {text}
    """

    # Generate response using Gemini
    response = model.generate_content(prompt)

    # Extract the JSON output
    json_output = response.text
    return json_output

def process_receipt(image_path):
    # step 1: Preprocess the image
    preprocessed_image = preprocess_image(image_path)

    # step 2: Extract text
    extracted_text = extract_text(preprocessed_image)

    # step 3: AI extraction
    json_output = ai_extract(extracted_text)

    return json_output

if __name__ == "__main__":
    image_path = "image.webp"
    result = process_receipt(image_path)
    print("Extracted JSON:")
    print(result)

