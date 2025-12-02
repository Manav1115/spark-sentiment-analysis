# Distributed Sentiment Analysis System
## Apache Spark + Flask + Web Interface

This project implements a distributed sentiment analysis system using Apache Spark for scalable text processing and classification. The system analyzes text data (tweets, reviews, comments) to classify sentiment as positive or negative using a Naive Bayes classifier deployed across a Spark cluster.


----------

## FEATURES

- **Distributed Computing**: Processes sentiment analysis across multiple worker nodes
- **Scalable Architecture**: 1 Master + 2 Worker nodes in standalone cluster mode
- **Machine Learning Pipeline**: PySpark MLlib with Naive Bayes classifier
- **REST API**: Flask-based backend for prediction requests
- **Web Interface**: User-friendly web frontend for real-time sentiment classification
- **Parallel Execution**: Demonstrates Big Data principles with distributed workloads
----------

## PROJECT STRUCTURE
```
Sentiment-Analysis-Distributed/
│
├── dataset.csv # Training dataset (text, label)
├── train_model.py # Model training script (run on Master)
├── backend.py # Flask API server & Web Controller
├── templates/ # HTML files for Web Interface
│ └── index.html # (Verify this folder exists in your repo)
├── sentiment_model/ # Trained model directory (generated)
├── README.md
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
- **Flask** (for backend & web serving)

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
HADOOP_HOME = C:\hadoop
```
Add to **PATH**
```
%SPARK_HOME%\bin
%SPARK_HOME%\sbin
%HADOOP_HOME%\bin
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


**Important** : Go to http://localhost:8080 in your browser. Note down the Master URL.
It will look like:  spark://192.168.x.x:7077.

### Step 3: Start Spark Worker (Laptop 2 / Laptop 3)

Run:

```
spark-class org.apache.spark.deploy.worker.Worker spark://<MASTER_IP>:7077
```
Terminal will look like this:
![03-Spark-Worker-Deploy](https://github.com/user-attachments/assets/15ed485b-6d42-41b6-94de-6fc5a080bcab)

**Example**: spark-class org.apache.spark.deploy.worker.Worker spark://192.168.1.5:7077


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

### Step 2: Access the Interface
Open your web browser and go to:

```bash
http://localhost:5000
```
(If accessing from a different laptop on the same network, use http://<MASTER_IP>:5000)


### Step 3 : Analyze Test
1. Enter text in the input field
2. Click "Analyze Sentiment"
3. View classification result (Positive/Negative)


### User Interface Preview
<img width="1303" height="874" alt="image" src="https://github.com/user-attachments/assets/01266235-a941-4b66-9adb-76d26958b7d1" />


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
│   Web Browser   │ ← User Input (HTML/CSS)
└────────┬────────┘
         │ HTTP Request
         ▼
┌─────────────────┐
│  Flask Backend  │ ← API & Web Server
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
| Frontend UI | HTML/CSS (Web Interface) |
| Data Processing | Pandas, NumPy |
| Language | Python 3.8+ |

----------

## TROUBLESHOOTING

### Issue: Workers not connecting to Master

**Solution**: 
- Verify Master URL is correct (spark://...).
- Check firewall settings (port 7077 should be open).
- Ensure all machines are on the same Wi-Fi network.

### Issue: Model training fails

**Solution**:
- Verify dataset.csv exists and has correct format.
- Windows Users: Ensure winutils.exe is present in %HADOOP_HOME%\bin.
- Ensure sufficient memory allocation in Spark.

### Issue: Flask API not accessible

**Solution**:
- Ensure backend.py is running.
- If accessing from another machine, ensure the Flask app is running with host='0.0.0.0'.
- Check if port 5000 is blocked by a firewall.
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

- **Real-time Streaming**: Implement Spark Streaming for live Twitter/X data analysis.
- **Advanced NLP Models**: Integrate BERT or RoBERTa transformers for higher accuracy.
- **Cloud Deployment**: Migrate cluster to AWS EMR or Azure HDInsight for auto-scaling.
- **Docker Support**: Containerize the Master and Worker nodes to simplify setup (removing the need for manual Java/Winutils installation).
- **Multi-class Classification**: Expand from Positive/Negative to include Neutral, Angry, and Joyful sentiments.
- **Advanced Dashboarding**: Replace basic UI with a rich analytics dashboard (using Streamlit or Dash) to visualize sentiment trends over time.
- **GPU Acceleration**: Leverage NVIDIA Spark RAPIDS for faster model training.

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
            
