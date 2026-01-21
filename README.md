# ProteinPredictor 🧬

ProteinPredictor is a machine learning-based tool designed to predict and visualize protein structures. The project utilizes CNN and Bi-LSTM models to process sequence data and offers a web-based visualization interface for 3D protein geometry.

## 🚀 Quick Start

### 1. Clone the repository

git clone [https://github.com/Vamsi-Krishna-Gottumukkala/ProteinPredictor.git](https://github.com/Vamsi-Krishna-Gottumukkala/ProteinPredictor.git)

cd ProteinPredictor

### 2. Download Datasets and Models

Due to file size limits, the trained weights and datasets are hosted in GitHub Releases.

Download ProteinPredictor.rar

Extract the .rar file into the project root.

Ensure your directory structure looks like this:

Plaintext
ProteinPredictor/

├── dataset/ # .csv and .npy files

├── models/ # .h5 weight files

├── src/ # Source code

├── web_app/ # Flask application

└── main.py

### 3. Install Dependencies

Make sure you have Python installed, then run:

pip install -r requirements.txt

### 4. Run the Application

To launch the web-based visualization tool:

python web_app/app.py

Then open your browser and navigate to http://127.0.0.1:5000.

### 🛠️ Features

Deep Learning Models: Uses CNN and Bi-LSTM architectures for sequence-to-structure prediction.

3D Visualization: Integrated viewer using JavaScript for interactive protein structure analysis.

Geometry Utils: Custom scripts for handling PDB files and distance matrices.

### 📂 Project Structure

dataset/: Contains distance data and PDB files used for training and testing.

models/: Pre-trained weights for the prediction models.

src/: Core logic including the prediction pipeline and quality assessment.

web_app/: Flask-based frontend and backend for the user interface.

### 📄 License

This project is for educational and research purposes.
