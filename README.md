# Animal Classifier App
Source code for image classification app developed by COSC320/COSC591 Group B for T1 2024. 

## The Problem 
Our client Lachlan is faced with the long manual task of having to manually label thousands of camera trap images for detecting 16 species of animals. This process takes a long time.

## The Solution
The solution for our problem for automating the wildlife camera trap images is developing a software system that integrates a fine-tuned YOLOv8 model with a user-friendly GUI. This software will facilitate the automated detection and identification of the 16 specified animal species from images stored on a local computer. The finished product will be a Python-based application that will reduce the effort and time required to analyse camera trap images.

## How it was made
The Animal Classifier App is built using a fine-tuned Ultralytics' YOLOv8 model, which has been trained on a custom dataset of 16 different species. The end-product is a Python-based application with the option of running a graphical user interface run locally on a web browser using Flask.

## Limitations
The model is only able to predict and detect species bases on the data it was trained on. Currently, this includes animals from any of the 16 species below:
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

## Installation (Linux/MacOS/Windows)
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

## Running the app on a VSCode devcontainer
1. Ensure the `devcontainer` plugin is installed in your VSCode and Docker is installed onto your machine
2. Open the folder with the code, which will prompt to reopen in the container. If not, get the `devcontainer` options in your command palette and select `reopen in container`. When building the container for the first time, expect to wait (~13GB of dependencies on a minimal Ubuntu OS)
3. Run:
```bash
python app.py
```
## Contribution
Limited to members of the COSC591 Group B only

## License
TBC
