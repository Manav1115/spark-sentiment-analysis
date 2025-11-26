# Distributed Sentiment Analysis System
## Apache Spark + Flask + Tkinter

This project implements a distributed sentiment analysis system using Apache Spark for scalable text processing and classification. The system analyzes text data (tweets, reviews, comments) to classify sentiment as positive or negative using a Naive Bayes classifier deployed across a Spark cluster.

----------

## FEATURES

- **Distributed Computing**: Processes sentiment analysis across multiple worker nodes
- **Scalable Architecture**: 1 Master + 2 Worker nodes in standalone cluster mode
- **Machine Learning Pipeline**: PySpark MLlib with Naive Bayes classifier
- **REST API**: Flask-based backend for prediction requests
- **Interactive UI**: Tkinter-based frontend for real-time sentiment classification
- **Parallel Execution**: Demonstrates Big Data principles with distributed workloads

----------

## PROJECT STRUCTURE
```
Sentiment-Analysis-Distributed/
│
├── dataset.csv              # Training dataset (text, label)
├── train_model.py           # Model training script (run on Master)
├── backend.py               # Flask API server
├── frontend.py              # Tkinter UI application
├── sentiment_model/         # Trained model directory (generated)
├── README.md
└── screenshots/
```

----------

## SYSTEM REQUIREMENTS

### Hardware Setup
- **Master Node** (Laptop A): Runs Spark Master + Model Training + Flask API
- **Worker Nodes** (Laptop B & C): Run Spark Workers for distributed processing
- **Minimum 8GB RAM** per machine recommended

### Software Prerequisites
- **Apache Spark 3.0+**
- **Python 3.8+**
- **JDK 8 or 11**
- **Flask** for backend API
- **Tkinter** for UI (usually comes with Python)

----------

## INSTALLATION

### Step 1: Install Dependencies
```bash
pip install pyspark flask pandas numpy
```

### Step 2: Apache Spark Setup

Download and extract Spark to:
```
C:\spark\
```

Set environment variables:
```
SPARK_HOME = C:\spark
PATH += C:\spark\bin; C:\spark\sbin
```

### Step 3: Verify Java Installation
```bash
java -version
```

----------

## DATASET

Create or use the provided `dataset.csv` with the following format:
```csv
text,label
"I love this product",1
"This is terrible",0
"Amazing experience",1
"Bad quality",0
"Great service",1
"Worst purchase ever",0
```

- **Label 1**: Positive sentiment
- **Label 0**: Negative sentiment

Dataset collected from various social media platforms and review systems.

----------

# SETTING UP SPARK CLUSTER (WINDOWS 11)

You can run Spark on multiple laptops:

-   Laptop 1 → Master
    
-   Laptop 2/3 → Worker nodes
    

### Step 1: Verify Java 

```
java -version

```
<img width="1103" height="169" alt="image" src="https://github.com/user-attachments/assets/87da6825-036e-4c9f-9b2f-d53b7331adff" />


### Step 2: Start Spark Master (Laptop 1)

Run:

```
spark-shell 
```

If spark is properly installed, you'll see something like this

![01_spark-shell](https://github.com/user-attachments/assets/d3d7c324-0dd4-4053-8ed0-ebe5faa91d33)


Run:

```
spark-class org.apache.spark.deploy.master.Master
```

Terminal will show the confirmation like this:-

![02-Spark-Master-Deploy](https://github.com/user-attachments/assets/4d2a760f-12a5-4c9d-afaa-fdf643e7813f)


Important : 
Note down the Master URL (something like spark://192.168.43.59:7077)

### Step 3: Start Spark Worker (Laptop 2 / Laptop 3)

Run:

```
spark-class org.apache.spark.deploy.worker.Worker spark://192.168.43.59:7077

```
Terminal will look like this:
![03-Spark-Worker-Deploy](https://github.com/user-attachments/assets/15ed485b-6d42-41b6-94de-6fc5a080bcab)


### Step 4: Verify Spark UI

Open:

```
http://localhost:8080

```

![SparkUI-running_job](https://github.com/user-attachments/assets/59c53a1a-6d92-4169-8a11-00d039ebaff2)

You should see:

-   Master node
    
-   Connected workers
    
-   CPU & memory resources
    

----------
## MODEL TRAINING

### Run Training Script (Only on Master Node)
```bash
python train_model.py
```

This script will:
1. Load dataset from `dataset.csv`
2. Preprocess text data
3. Train Naive Bayes classifier using PySpark MLlib
4. Distribute training across worker nodes
5. Save trained model to `sentiment_model/` directory

**Verify model creation**:
```bash
ls sentiment_model/
```

----------

## RUNNING THE APPLICATION

### Step 1: Start Flask Backend (Master Node)
```bash
python backend.py
```

Backend will run at:
```
http://10.10.31.111:5000
```

The Flask API communicates with the Spark cluster for predictions.

### Step 2: Launch Frontend UI (Any Machine)
```bash
python frontend.py
```

**Usage**:
1. Enter text in the input field
2. Click "Analyze Sentiment"
3. View classification result (Positive/Negative)

----------

## API ENDPOINTS

### POST `/predict`

**Request**:
```json
{
  "text": "This product is amazing!"
}
```

**Response**:
```json
{
  "sentiment": "Positive",
  "confidence": 0.85
}
```

----------

## ARCHITECTURE
```
┌─────────────────┐
│  Tkinter UI     │ ← User Input
└────────┬────────┘
         │ HTTP Request
         ▼
┌─────────────────┐
│  Flask Backend  │ ← Prediction API
└────────┬────────┘
         │ Spark Job
         ▼
┌─────────────────┐
│  Spark Master   │ ← Job Scheduling
└────────┬────────┘
         │ Distribute
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│Worker 1│ │Worker 2│ ← Parallel Processing
└────────┘ └────────┘
```

----------

## TECHNOLOGY STACK

| Component | Technology |
|-----------|-----------|
| Distributed Computing | Apache Spark 3.0 (Standalone Cluster) |
| Machine Learning | PySpark MLlib (Naive Bayes) |
| Backend API | Flask |
| Frontend UI | Tkinter |
| Data Processing | Pandas, NumPy |
| Language | Python 3.8+ |

----------

## TROUBLESHOOTING

### Issue: Workers not connecting to Master

**Solution**: 
- Verify Master URL is correct
- Check firewall settings (port 7077 should be open)
- Ensure all machines are on the same network

### Issue: Model training fails

**Solution**:
- Verify `dataset.csv` exists and has correct format
- Check Spark worker resources in UI
- Ensure sufficient memory allocation

### Issue: Flask API not accessible

**Solution**:
- Update IP address in `backend.py` to Master node's IP
- Check if port 5000 is available
- Verify Spark context is initialized

----------

## BASIC COMMANDS REFERENCE

### Check Spark Processes
```bash
jps
```

### Stop Spark Master
```bash
spark-class org.apache.spark.deploy.master.Master --stop
```

### Stop Spark Worker
```bash
spark-class org.apache.spark.deploy.worker.Worker --stop
```

### View Spark Logs
```bash
tail -f $SPARK_HOME/logs/spark-*.out
```

----------

## PROJECT TEAM

**Submitted by:**
- Manav Anand (221210065)
- Shubham Gupta (221210101)
- Tarang Srivastava (221210109)
- Vanshika Garg (221210119)

**Branch**: CSE  
**Semester**: 7th  
**Course**: Big Data Analytics (CSBB 422)  
**Guide**: Dr. Priten Savaliya

**Institution**: National Institute of Technology, Delhi

----------

## FUTURE ENHANCEMENTS

- Real-time streaming sentiment analysis using Spark Streaming
- Integration with larger NLP models (BERT, Transformers)
- Cloud deployment (AWS EMR, Azure HDInsight)
- Support for multi-class sentiment classification
- GPU-accelerated processing for deep learning models
- Web-based dashboard using Streamlit
- Enterprise-scale analytics pipeline integration

----------

## CONCLUSION

This project demonstrates the practical application of distributed computing in NLP tasks. By leveraging Apache Spark's parallel processing capabilities, the system showcases how machine learning workloads can scale across multiple nodes for improved performance and efficiency. The end-to-end implementation—from dataset preparation to live prediction through an interactive UI—provides a foundation for building production-grade sentiment analysis systems in Big Data environments.

----------

## LICENSE

This project is submitted as part of academic coursework at NIT Delhi.

----------

## ACKNOWLEDGEMENTS

Special thanks to:
- Dr. Priten Savaliya for guidance and support
- NIT Delhi for providing resources
- Apache Spark and open-source community
- Social media platforms for dataset sources

----------

## CONTACT

For questions or issues, please contact the project team through the institution.
