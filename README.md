# alphabet-detection
This project aims to detect and recognize sign language alphabets using computer vision techniques. Currently, it supports the detection of five alphabets: A, B, C, L, and Y. Leveraging libraries such as MediaPipe and OpenCV, the system processes image input from user and identifies hand gestures corresponding to these alphabets.
# Project Structure
1. app.py: Flask file responsible for running the web application.
2. model.p: Trained model file used for inference in the Flask application.
3. model.zip: Contains Python files for collecting images, preparing data, training the model, and getting inferences from the trained model. It includes a data folder with all collected data.
# Setup Instructions
1. Clone the repository

Bash : git clone https://github.com/navya-jain-cs/alphabet-detection

2. Navigate to the project directory:

Bash  : cd sign-language-detection

3. Install the required dependencies mentioned in the requirements.txt file:

Bash : pip install -r requirements.txt

4. Run the Flask application using the following command:

Bash : python app.py

Access the web application through the provided URL.
# Usage
1. Upload an image containing a hand gesture representing one of the supported alphabets (A, B, C, L, Y).
2. Click the "Submit" button to process the uploaded image.
3. View the detected alphabet on the screen.
# Contributions
Contributions and improvements are welcome! Feel free to fork the repository, make your changes, and submit a pull request. Whether it's adding support for additional alphabets, improving accuracy, or enhancing the user interface, your contributions are valuable to the project.

# Note
This project serves as a demonstration of sign language alphabet detection using computer vision techniques. While it currently supports a limited set of alphabets, the architecture can be extended to recognize more gestures and potentially be integrated into assistive technologies for the hearing-impaired community.
