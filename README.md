# AI-Powered Payment Fraud Detection Engine  

## Overview  
This project is an AI-driven payment fraud detection system that leverages a deep learning autoencoder to identify anomalous transactions. It is built using Python with TensorFlow for the deep learning model, scikit‑learn for data preprocessing, and Pandas/NumPy for data manipulation. The autoencoder is trained solely on normal transactions from the Kaggle Credit Card Fraud Detection dataset so that it learns the inherent patterns of genuine payment behavior. During inference, each new transaction is passed through the model, and its reconstruction error is computed. Transactions with errors exceeding a threshold (determined from the training error distribution) are flagged as suspicious. The project simulates real-time streaming of transactions and uses a Tkinter-based graphical user interface to display results interactively. This solution demonstrates the potential of deep learning for fraud detection and provides a foundation that can be extended for cloud deployment, integration with streaming platforms like Apache Kafka, and containerized microservices architectures on platforms such as AWS, Google Cloud, or Azure.

![Image](https://github.com/user-attachments/assets/89a45d8c-def5-4273-a636-8d072bd98622)

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

https://github.com/user-attachments/assets/93bf90fe-f640-4d46-ad57-420a37c2bd58

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

