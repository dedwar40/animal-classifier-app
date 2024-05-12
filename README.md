# Automated Animal Detection in Camera Trap Imagery Using Machine Learning

# Animal Classifier App
Source code for image classification app developed by COSC320/COSC591 Group B for T1 2024. 

## The Problem 
Camera traps capture images of animals in their natural habitats using motion sensors and offer a cost-efficient and non-invasive way to monitor wildlife species. However, considerable time is required to manually go through each image and determine whether an animal is present, and if so, which species the animal is.

## The Solution
This project developed an automated camera trap image processing system, leveraging machine learning to identify 16 species of animals in camera trap images. This system is a Python-based application that integrates a fine-tuned Ultralytics YOLOv8 model with a user friendly graphical user interface (GUI) that runs locally on a web browser using Flask. 

In total 5 models were developed based on the 5 YOLOv8 models provided by Ultralytics, with the following results on held-out validation data:

- yolov8n_16species model correctly identified with 96.5% precision and 95.3% recall
- yolov8s_16species model correctly identified with 97.2% precision and 96.4% recall
- yolov8m_16species model correctly identified with 97.3% precision and 96.2% recall
- yolov8l_16species model correctly identified with 97.4% precision and 98.7% recall
- yolov8x_16species model correctly identified with 97.2% precision and 96.8% recall

Model weights for the two smallest models can be found in the 'weights' folder.

## How it was made
The Animal Classifier App was fine-tuned on a custom dataset of 16 different species, containing 1,000 images of each species, including:

1. Rabbit
2. Brush-tailed Rock Wallaby
3. Spotted-tailed Quoll
4. Swamp Wallaby
5. Eastern Grey Kangaroo
6. Echidna
7. Red-necked Wallaby
8. Bandicoot
9. Brushtail Possum
10. Superb Lyrebird
11. Kookaburra
12. Magpie
13. Currawong
14. Cat
15. Fox
16. Goat

Bounding boxes were placed around animals in each image, and the coordinates of the bounding boxes were then used to fine tune the model, along with the images themselves.

## Limitations
The model is only able to classify animal species from those listed above.

Larger models tend to provide better results on new data, however, they take longer to run compared to smaller models.

## Installation (Linux/MacOS/Windows)
These steps have been tested for MacOS, Linux and Windows(11) operating systems. 

1. Ensure Python 3 is installed on your local machine, alongside `pip` and `venv`
2. Clone the GitHub repository onto your local machine using `git clone` OR download the zip file and uncompress the contents
3. Create + activate a virtual environment for the project
```bash
python3 -m venv .venv
source .venv/bin/activate
```
4. Install all dependecies listed in `requirements.txt`. This can be done with:
```bash
pip install -r requirements.txt
```
## Usage
To run the app directly on your machine:
```bash
python app.py
```
To run the Flask app on your local machine, run:
```bash
python -m flask run
```
`Ctrl + Click` on the URL provided in the terminal to open up the Flask app in your web browser.

## Contribution
Limited to members of the COSC591 Group B only

## License
TBC
