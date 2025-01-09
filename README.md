# CNN-LSTM for EMG Gesture Classification

## 1. Project Overview
This repository contains a **CNN-LSTM-based** approach to classifying **Electromyography (EMG)** signals. EMG signals measure the electrical activity of muscles and are widely used for **gesture recognition**, prosthetic control, and other biomedical applications. By combining **Convolutional Neural Networks (CNN)** with **Long Short-Term Memory (LSTM)** layers, this model captures both **spatial and temporal dependencies** in multi-channel EMG data. For a more detailed description, see the report: **EMG_Gesture_Report.pdf**.

---

## 2. Key Features
- **Multi-Channel Input**: Handles multiple EMG channels.  
- **CNN+LSTM Architecture**: CNN extracts spatial features, while LSTM processes temporal patterns.  
- **Configurable Hyperparameters**: Easily adjust hidden sizes, number of layers, dropout rates, and learning rate.  
- **Training & Evaluation Pipeline**: Includes training loop, validation, and testing with metrics (Accuracy, F1-Score, Precision, Recall).  
- **Visualization**: Plots training loss, validation accuracy, and confusion matrix for model performance analysis.

---

## 3. Project Structure
```text
.
├── data/
│   └── EMG-data.csv             # CSV file with EMG signals and labels
├── src/
│   ├── dataset.py               # Custom PyTorch Dataset for EMG data
│   ├── model.py                 # CNN-LSTM model definition
│   ├── train.py                 # Training loop, validation, and logging metrics
│   └── evaluate.py              # Testing & evaluation script
├── report/
│   └── EMG_Gesture_Report.pdf   # Final report
└── README.md                    # This README file

```
---

## 4. Dataset Description

This project uses raw EMG data collected via a MYO Thalmic bracelet. The device is worn on a user’s forearm and communicates over Bluetooth to a PC. 
The bracelet has eight equally spaced EMG sensors around the forearm that simultaneously acquire myographic signals.
A total of 36 subjects participated, each performing a series of static hand gestures twice. Each gesture lasted for approximately 3 seconds, with a 3-second pause between gestures.

Columns in the Dataset

Each row in the raw EMG data file contains 11 columns:
	1.	Time (in ms): Timestamp for the recorded signal.
	2.	Channel 1: Raw EMG signal from the first sensor.
	3.	Channel 2: Raw EMG signal from the second sensor.
	4.	Channel 3: Raw EMG signal from the third sensor.
	5.	Channel 4: Raw EMG signal from the fourth sensor.
	6.	Channel 5: Raw EMG signal from the fifth sensor.
	7.	Channel 6: Raw EMG signal from the sixth sensor.
	8.	Channel 7: Raw EMG signal from the seventh sensor.
	9.	Channel 8: Raw EMG signal from the eighth sensor.
	10.	Class: Integer label representing the gesture performed:
          	•	0: Unmarked data
        	•	1: Hand at rest
         	•	2: Hand clenched in a fist
        	•	3: Wrist flexion
        	•	4: Wrist extension
        	•	5: Radial deviation
         	•	6: Ulnar deviation
        	•	7: Extended palm (all subjects did not perform this gesture)

 11.	Label (added manually): Identifies the subject who performed the gesture. There are 36 distinct subjects in total, each having performed 7 gestures twice.
	•	Data Source: The EMG-data.csv file contains multi-channel EMG recordings, associated labels (gestures or classes), and optional timestamps.
	•	Preprocessing: Data is normalized using StandardScaler; additional feature engineering (e.g., windowing, filtering) can be applied if required.
	•	Splits: Typically split into training, validation, and testing sets (70/15/15) using train_test_split.

## 5. Requirements
	•	Python 3.11+
	•	PyTorch (>= 2.1)
	•	scikit-learn (>= 0.24)
	•	pandas, numpy, matplotlib
 ---

## 6. Results / Model Performance:
	•	Validation Accuracy: ~92%
	•	Validation F1-Score: ~88%
	•	Test Accuracy: ~90–92%
 ---

## 7. Future Work
	•	Hyperparameter Tuning: Automate search with libraries like Optuna or Ray Tune.
	•	Data Augmentation: Use overlapping windows, synthetic augmentation to handle limited data.
	•	Advanced Architectures: Experiment with GRUs, Transformers, or attention mechanisms.
	•	Real-Time Inference: Optimize latency for real-time gesture control.
 ---

## 8. Contact
        For questions, suggestions, or collaborations, please contact:
	•	Name: Aleksei KUZNETSOV
	•	Email: aleksei.kuznetsov@protonmail.com
 ---

## Acknowledgements:
        •	UCI Machine Learning Repository: We extend our gratitude to the team and researchers who collected and shared this dataset.
	•	Contributors: Thank you to all the volunteers (36 subjects) who provided EMG recordings for this research.
 ---

For more information, refer to the original dataset page:
https://archive.ics.uci.edu/ml/datasets/EMG+data+for+gestures
