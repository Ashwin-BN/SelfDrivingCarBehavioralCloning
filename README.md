# SelfDrivingCarBehavioralCloning

## Project Overview

This project implements an end-to-end deep learning model for behavioral cloning to predict steering angles in autonomous driving. Using NVIDIA’s CNN architecture, the model learns from driving images and steering data to control a simulated car.

The trained model integrates with the Udacity Self-Driving Car Simulator to autonomously steer the car in real-time.

---

## Approach & Challenges

### Approach

* **Data Processing:** Driving logs are loaded, image paths normalized, and invalid samples removed. To reduce dataset bias, straight-driving samples are downsampled while turning samples are fully retained.
* **Image Preprocessing:** Images are cropped, converted to YUV color space, blurred, resized, and normalized to improve learning.
* **Data Augmentation:** Random horizontal flips with steering angle inversion augment the dataset, enhancing model robustness.
* **Model:** NVIDIA’s CNN architecture is used, combining convolutional layers with fully connected layers to regress steering angles.
* **Training:** Early stopping is employed to prevent overfitting, and the model is saved after training.
* **Simulation:** The trained model is deployed via a Flask + SocketIO server that communicates with the Udacity simulator for real-time driving.

---

### Challenges & Solutions

1. **Collecting Good Driving Data**  
   When I first collected driving data by manually controlling the car, my driving was inconsistent and rough. This meant the model learned incorrect steering patterns and didn’t drive well in the simulator. I had to redo data collection multiple times, making sure to drive smoothly and accurately to produce clean, high-quality training data.

2. **Overfitting During Training**  
   Initially, the model performed well on training data but poorly on new, unseen data. This overfitting was fixed by adding data augmentation (random flips with steering angle inversion) and early stopping, which prevented the model from training for too long and memorizing the dataset instead of learning patterns.

3. **TensorFlow & Python Version Conflicts**  
   I ran into compatibility issues between TensorFlow and my Python virtual environment. Some TensorFlow features weren’t supported in my current setup. I solved this by adjusting my Python version and downgrading TensorFlow to a stable version that worked with my dependencies.

4. **Handling Large Dataset Files**  
   The `IMG` folder with collected driving images is very large and can’t be pushed to GitHub normally. I used **Git Large File Storage (Git LFS)** to track and store these big files without hitting repository size limits.

**Detailed training report:** [REPORT.md](./REPORT.md)

---

## Demo Video

[![SelfDrivingCarBehavioralCloning Demo](https://img.youtube.com/vi/Zfg080-YTvY/maxresdefault.jpg)](https://youtu.be/Zfg080-YTvY)

---

## Environment Setup & Dependencies

### Prerequisites

* Python 3.7 or newer
* Git (optional, for cloning the repo)
* Git LFS (optional, for handling large files like the `IMG` folder)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ashwin-BN/SelfDrivingCarBehavioralCloning.git
   cd SelfDrivingCarBehavioralCloning
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Install Git LFS and pull large files:

   ```bash
   git lfs install
   git lfs pull
   ```

---

## Running the Project

### Training the Model

1. Ensure the dataset CSV (`driving_log.csv`) and images folder (`IMG/`) are in the project root.
2. Run the training script:

   ```bash
   python train_model.py
   ```
3. The training process will:

   * Display and save steering angle distribution histograms before and after balancing.
   * Train the NVIDIA CNN model with early stopping.
   * Save the trained model as `model.h5`.
   * Plot and save training/validation loss curves (`training_plot.png`).

### Running the Simulator with the Trained Model

1. Make sure `model.h5` is present.
2. Start the simulation server:

   ```bash
   python TestSimulation.py
   ```
3. Open the Udacity Self-Driving Car Simulator.
4. Connect the simulator to the server on port `4567`.
5. The car should drive autonomously using your trained model.

---

## Data Collection and Simulator Testing

For detailed instructions on **how to collect your own driving data using the Udacity Simulator** and **how to test your trained model**, please see [DATA\_COLLECTION.md](./DATACOLLECTION.md).

---

## Project Structure

```
.
├── IMG/                          # Driving images (tracked with Git LFS)
├── driving_log.csv               # Driving data CSV file
├── train_model.py                 # Model training script
├── TestSimulation.py              # Simulator integration script
├── model.h5                       # Trained model (output)
├── steering_before_balancing.png  # Histogram before data balancing
├── steering_after_balancing.png   # Histogram after data balancing
├── training_plot.png              # Training and validation loss plot
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── REPORT.md                     # Model training and data distribution report
└── DATA_COLLECTION.md             # Data collection and testing guide
```

---

## Contact & References

* Developed by: **Ashwin B N**
* Inspired by NVIDIA’s end-to-end learning paper: [https://arxiv.org/abs/1604.07316](https://arxiv.org/abs/1604.07316)
* Udacity Simulator GitHub: [https://github.com/udacity/self-driving-car-sim](https://github.com/udacity/self-driving-car-sim)

---

*Feel free to raise an issue if you encounter any problems.*
