# Spark MLlib Spam Classifier (Scala)

## 📘 Overview
This project trains a simple spam classifier using **Apache Spark MLlib** in Scala.  
It uses tokenization, stop-word removal, TF‑IDF weighting, and a **Linear SVM classifier** to classify SMS messages into *spam* or *non‑spam* categories.  
Cross‑validation is performed to tune hyperparameters and evaluate model performance.

---

## 🧩 Folder Structure
```
spam_classifier/
├── data/
│ ├── training_spam.txt
│ ├── training_nospam.txt
│ ├── testing_spam.txt
│ └── testing_nospam.txt
└── SpamClassifier.scala
```
---

## ⚙️ Setup Instructions

1. **Make sure Java, Scala, and Spark are installed** in your WSL environment:
   ```bash
   java -version
   scala -version
   spark-shell
2. Navigate to the project folder:
```bash
   cd ~/projects/spam_classifier
```
3. Compile and run the Scala program: 
```bash
scalac -classpath "$SPARK_HOME/jars/*" SpamClassifier.scala
jar cf SpamClassifier.jar SpamClassifier*.class
spark-submit --class SpamClassifier --master local[*] SpamClassifier.jar
 ```

## 🚀 Output Example
After training and evaluation, Spark prints metrics such as:
 ```
---- Training ----
F1‑Score = 1.0
Accuracy = 1.0

---- Testing ----
F1‑Score = 0.99
Accuracy = 0.99
Confusion Matrix:
Spam NotSpam
2.0 0.0
0.0 2.0
 ```

## 👤 Author
Manav Anand - 221210065

Shubham Gupta - 221210101

Tarang Srivastava - 221210109

Vanshika Garg - 221210119
