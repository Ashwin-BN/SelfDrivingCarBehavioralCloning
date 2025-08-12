# Data Collection and Simulator Testing Guide

## 1. Introduction

This guide explains how to collect your own driving data (images and steering angles) using the Udacity Self-Driving Car Simulator. It also covers how to test your trained model with the simulator.

---

## 2. Collecting Driving Data

### Step 1: Download the Simulator

* Download the Udacity Self-Driving Car Simulator for your platform from the [official release page](https://github.com/udacity/self-driving-car-sim/releases).
* Extract the downloaded archive.

### Step 2: Launch the Simulator

* Run the simulator executable (`beta_simulator.exe` on Windows or equivalent for Mac/Linux).
* The main window will open, showing the mode selection screen.

### Step 3: Select Training Mode

* Choose **Training Mode** to manually drive the car and collect data.
* This mode records your driving inputs (steering angle, throttle) and images from the car’s cameras.

### Step 4: Configure Recording Path

* From the top menu, click the **Recording** option.
* Select a directory where data will be saved (this will generate an `IMG` folder and a `driving_log.csv` file).
* Click **Select** to confirm.

### Step 5: Drive and Record Data

* Use keyboard (and mouse for smoother steering) to drive the car along the track.
* It's recommended to:

  * Drive multiple laps (about 5 forward, 5 reverse) for balanced data.
  * Use the **leftmost path** for easier data collection.
  * Steer smoothly to get varied but realistic data.
* Your driving images will be saved in the `IMG` folder, and the `driving_log.csv` records the steering angle and other telemetry.

### Step 6: End Recording

* After finishing the laps, stop recording.
* Verify the `IMG` folder and `driving_log.csv` exist in your chosen directory.

---

## 3. Using Your Data for Training

* Copy the `IMG` folder and `driving_log.csv` to your project root.
* Run the training script:

```bash
python train_model.py
```

* This script will load your data, balance it, augment, and train the NVIDIA CNN model.
* It saves the trained model as `model.h5`.

---

## 4. Testing Your Trained Model in Simulation

### Step 1: Start the Simulation Server

Run the Flask + SocketIO server that loads your trained model:

```bash
python TestSimulation.py
```

### Step 2: Launch the Udacity Simulator

* Open the Udacity Self-Driving Car Simulator.
* Choose **Autonomous Mode**.

### Step 3: Connect to the Server

* The simulator automatically connects to the server on port `4567`.
* Your trained model will start receiving images and sending steering/throttle commands.

### Step 4: Watch the Car Drive

* The car should now drive autonomously based on your trained model.
* Monitor the server console for telemetry info.

---

## 5. Tips & Best Practices

* Ensure you drive carefully for data collection.
* Collect enough data with diverse steering angles, especially turns.
* Use smooth and continuous steering to avoid noisy labels.
* Balance your dataset (reduce excessive straight driving samples) to improve training.

---

## 6. Additional Resources

* Udacity Simulator GitHub: [https://github.com/udacity/self-driving-car-sim](https://github.com/udacity/self-driving-car-sim)
* NVIDIA End-to-End Learning Paper: [https://arxiv.org/abs/1604.07316](https://arxiv.org/abs/1604.07316)

---

If you encounter any issues or need help, feel free to open an issue.