from flask import Flask, render_template, request
from flask import render_template
from pathlib import Path
from werkzeug.utils import secure_filename
import os
import shutil
from classifer import get_img_classification

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
IMG_COUNTER = 0
SUBFOLDER = ''

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

def ensure_folders_exist():
    # Create uploads folder if it doesn't exist
    uploads_folder = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
    if not os.path.exists(uploads_folder):
        os.makedirs(uploads_folder)
        print(f"Created '{uploads_folder}' folder.")

    # Create results folder if it doesn't exist
    results_folder = os.path.join(app.static_folder, app.config['RESULTS_FOLDER'])
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        print(f"Created '{results_folder}' folder.")

ensure_folders_exist()

IMAGES = os.listdir(os.path.join(app.static_folder, "results"))

@app.route("/")
def hello_world():
    return render_template(
        "index.html"
    )

#
# This function allows the user to upload multiple files that
# will save to the /uploads folder.
# From here we can loop through the images to process them.
# We will need to clean the uploads folder after processing the images.
#
@app.route('/upload', methods=["POST"])
def uploadFiles():
    if request.method == "POST":
        basepath = os.path.dirname(__file__)
        files = request.files.getlist('files')
        if 'files' in request.files:
            try:
                for f in files:
                    filename = secure_filename(f.filename)
                    filepath = os.path.join(basepath, 'uploads', filename)
                    f.save(filepath)
                return render_template("index.html", ready=True)
            except FileNotFoundError:
                print('File not found.')
                return render_template("index.html", empty=True)

#
# This function will get the value of the bounding box and call
# the methods for image prediction.
# If the bounding box is selected 'boundingBox' will equal "on", else None
# It will need to process all of the images in the uploads folder,
# save them to the /results folder
#
@app.route('/predict', methods=["POST"])     
def predict():
    if request.method == "POST":
        # run image prediction and save to results
        source_directory = os.path.join(os.path.dirname(__file__), 'uploads')
        if(len(os.listdir(source_directory)) == 0):
            return render_template("index.html", empty=True)
        else:
            # Iterate through subdirectories and move files to uploads folder
            for root, dirs, files in os.walk(source_directory):
                for file in files:
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(source_directory, file)
                    shutil.move(src_path, dst_path)
            results = get_img_classification('src/models/best.pt', source_directory, app.static_folder, "results")
            # get statistics
            stats_data = {}
            for result in results.get('boxes'):
                animal = str(result.get('animal'))
                confidence_score = str(result.get('confidence_score'))
                box_coordinates_normalised = str(result.get('box_coordinates_normalised'))
                image_filename = str(result.get('image_filename'))
                
                if image_filename not in stats_data:
                    stats_data[image_filename] = []
                
                stats_data[image_filename].append(
                    {
                        "animal": animal,
                        "confidence_score": confidence_score,
                        "box_coordinates_normalised": box_coordinates_normalised,
                        "image_filename": image_filename
                    }
                )
            
            for image_filename, stats in stats_data.items():
                save_path = os.path.join(app.static_folder, "results", stats[0]["animal"])
                os.makedirs(save_path, exist_ok=True)
                
                filename = os.path.join(save_path, image_filename.split('.')[0] + ".txt")
                with open(filename, "w") as file:
                    for stat in stats:
                        file.write(
                            f"{stat['animal']}\n"
                            f"{stat['confidence_score']}\n"
                            f"{stat['box_coordinates_normalised']}\n"
                            f"{stat['image_filename']}\n"
                        )
                    file.close()
            
            # get list of all subdirectories in /results
            results_dir = os.path.join(app.static_folder, "results")
            subfolders = [
                folder for folder in os.listdir(results_dir) 
                if os.path.isdir(os.path.join(results_dir, folder)) and not folder.endswith('.log')
    ]
            # clear the uploads folder
            [os.remove(f) if f.is_file() else shutil.rmtree(f) 
             for f in Path(os.path.join(os.path.dirname(__file__), 'uploads')).rglob("*") 
             if f.is_file() or os.path.isdir(f)]
            
            return render_template("index.html", subfolders=subfolders, complete=True)

#
# This image will display the subfolders created during image prediction
#
@app.route('/view_images', methods=["POST"])
def view():
    # Get list of all subdirectories in /results excluding .log files
    results_dir = os.path.join(app.static_folder, "results")
    subfolders = [
        folder for folder in os.listdir(results_dir) 
        if os.path.isdir(os.path.join(results_dir, folder)) and not folder.endswith('.log')
    ]
    return render_template("index.html", subfolders=subfolders)

#
# Display images in selected subfolder
#
@app.route('/display', methods=["POST"])
def display():
    subfolder = request.form['subfolder']
    # save information for displaying images
    global IMAGES 
    IMAGES = os.listdir(os.path.join(app.static_folder, "results/" + subfolder))
    global IMG_COUNTER
    IMG_COUNTER = 0
    global SUBFOLDER
    SUBFOLDER = subfolder
    current_img = "/static/results/" + SUBFOLDER + "/" + IMAGES[IMG_COUNTER]
    while(
        not current_img.lower().endswith(('.png', '.jpg', '.jpeg', 'tiff', '.bmp', '.gif'))
        and ((IMG_COUNTER + 1) != (len(IMAGES) - 1))
    ):
        IMG_COUNTER += 1
        current_img = "/static/results/" + SUBFOLDER + "/" + IMAGES[IMG_COUNTER]
    
    stats = display_stats(IMAGES[IMG_COUNTER].split('.')[0], SUBFOLDER)
    
    if stats is not None:
        return render_template("index.html", 
                               current_img=current_img,
                               stats=stats)
    else:
        return render_template("index.html", current_img=current_img)

#
# Navigates to the previous image, keeping track of place in results folder
#
@app.route('/previous_img', methods=["POST"])
def previous():
    global IMG_COUNTER
    global IMAGES
    if IMG_COUNTER != 0:
        IMG_COUNTER = IMG_COUNTER - 1
    
    current_img = "/static/results/" + SUBFOLDER + "/" + IMAGES[IMG_COUNTER]
    while(
        not current_img.lower().endswith(('.png', '.jpg', '.jpeg', 'tiff', '.bmp', '.gif'))
        and IMG_COUNTER != -1
    ):
        IMG_COUNTER -= 1
        current_img = "/static/results/" + SUBFOLDER + "/" + IMAGES[IMG_COUNTER]
    
    stats = display_stats(IMAGES[IMG_COUNTER].split('.')[0], SUBFOLDER)
    
    if stats is not None:
        return render_template("index.html", 
                               current_img=current_img,
                               stats=stats)
    else:
        return render_template("index.html", current_img=current_img)

#
# Navigates to the next image, keeping track of place in results folder
#
@app.route('/next_img', methods=["POST"])
def next():
    global IMG_COUNTER
    global IMAGES
    if (SUBFOLDER != "unknown"):
        if IMG_COUNTER != len(IMAGES) - 2:
            IMG_COUNTER = IMG_COUNTER + 1
    else:
        if IMG_COUNTER != len(IMAGES) - 1:
            IMG_COUNTER = IMG_COUNTER + 1
    print(IMG_COUNTER)
    
    current_img = "/static/results/" + SUBFOLDER + "/" + IMAGES[IMG_COUNTER]
    while(
        not current_img.lower().endswith(('.png', '.jpg', '.jpeg', 'tiff', '.bmp', '.gif'))
        and IMG_COUNTER != (len(IMAGES) - 1)
    ):
        IMG_COUNTER += 1
        print("Found a non-image file, skipping to img: ")
        print(IMG_COUNTER)
        current_img = "/static/results/" + SUBFOLDER + "/" + IMAGES[IMG_COUNTER]
    
    stats = display_stats(IMAGES[IMG_COUNTER].split('.')[0], SUBFOLDER)
    
    if stats is not None:
        return render_template("index.html", 
                               current_img=current_img,
                               stats=stats)
    else:
        return render_template("index.html", current_img=current_img)

def display_stats(image_filename, subfolder):
    filename = os.path.join(app.static_folder, "results", subfolder, f"{image_filename}.txt")
    if Path(filename).exists():
        with open(filename) as file:
            lines = [line.rstrip() for line in file]
        
        stats = []
        for i in range(0, len(lines), 4):
            stats.append({
                "animal": lines[i],
                "confidence_score": lines[i+1],
                "box_coordinates_normalised": lines[i+2],
                "image_filename": lines[i+3]
            })
        return stats
    return None