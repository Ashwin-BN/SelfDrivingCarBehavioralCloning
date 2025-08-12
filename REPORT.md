# SelfDrivingCarBehavioralCloning – Training Report

This document summarizes the data distribution, model training process, and final results.

---

## 1. Steering Angle Distribution

A balanced dataset is crucial to avoid the model always predicting “straight” steering.  
Below are the histograms **before** and **after** balancing the dataset.

### Before Balancing
![Steering Angle Histogram – Before Balancing](./steering_before_balancing.png)

### After Balancing
![Steering Angle Histogram – After Balancing](./steering_after_balancing.png)

---

## 2. Training Loss Plot

The plot below shows the model’s training and validation loss over epochs.  
Early stopping was used to prevent overfitting.

![Training vs Validation Loss](./training_plot.png)

---

## 3. Observations

* The original dataset had a strong bias toward straight driving.
* After balancing, the distribution became more even, improving model generalization.
* Training loss decreased steadily, and validation loss stabilized early, showing good generalization.

---

## 4. Model Performance

* Successfully navigated the Udacity Simulator’s first track without leaving the road.
* Handles turns and curves much better after balanced training data.
* Minimal steering jitter compared to early model versions.
