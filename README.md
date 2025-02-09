# AI-Powered Payment Fraud Detection Engine  

## Overview  
This project is an AI-driven payment fraud detection system that utilizes a deep learning autoencoder to identify fraudulent transactions based on anomaly detection. It processes real-time payment data and flags suspicious transactions based on reconstruction error thresholds.  
https://github.com/user-attachments/assets/7c70d8e7-6181-4249-9043-b9e6557d9264

## How It Works  
1. **Data Processing**  
   - The system uses the Kaggle Credit Card Fraud Detection dataset.  
   - Data is preprocessed to normalize features and remove irrelevant fields.  

2. **Model Training**  
   - A deep learning autoencoder is trained using TensorFlow/Keras.  
   - The model learns patterns from non-fraudulent transactions.  
   - Anomaly detection is performed by measuring reconstruction errors.  

3. **Real-Time Transaction Simulation**  
   - Transactions are streamed into the system in real-time.  
   - Each transaction is processed by the trained model.  
   - If the reconstruction error exceeds a predefined threshold, it is flagged as fraudulent.  

4. **User Interface**  
   - Built with Tkinter for a simple desktop-based GUI.  
   - Displays real-time transaction statuses and fraud alerts.  
![Image](https://github.com/user-attachments/assets/e454ad09-53e6-48ec-8b59-f273f07db1c1)

## Tech Stack  
- **Programming Language:** Python  
- **Machine Learning Framework:** TensorFlow/Keras  
- **GUI Framework:** Tkinter  
- **Data Handling:** Pandas, NumPy  
- **Model Deployment:** Pickle (for model storage)  

## How to Use  
1. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```  
2. **Run the Application**  
   ```bash
   python main.py
   ```  
3. **Interact with the GUI**  
   - View real-time transaction processing.  
   - Fraudulent transactions will be flagged automatically.  

## Dataset  
- The system uses the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).  

## Future Enhancements  
- Integrating a more advanced deep learning model (e.g., LSTMs or Transformers).  
- Adding cloud deployment with Flask/FastAPI for API-based fraud detection.  
- Implementing Kafka for real-time transaction streaming at scale.  

