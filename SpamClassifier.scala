import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.{Pipeline}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions._

object SpamClassifier {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder
      .appName("Spark Spam Classifier")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    val spam_train = spark.read.textFile("./data/training_spam.txt").toDF("text").withColumn("label_category", lit("spam"))
    val nospam_train = spark.read.textFile("./data/training_nospam.txt").toDF("text").withColumn("label_category", lit("nospam"))
    val spam_test = spark.read.textFile("./data/testing_spam.txt").toDF("text").withColumn("label_category", lit("spam"))
    val nospam_test = spark.read.textFile("./data/testing_nospam.txt").toDF("text").withColumn("label_category", lit("nospam"))

    val train_df = spam_train.union(nospam_train)
    val test_df = spam_test.union(nospam_test)

    println(s"Spam training count: ${spam_train.count()}")
    println(s"No spam training count: ${nospam_train.count()}")
    println(s"Spam testing count: ${spam_test.count()}")
    println(s"No spam testing count: ${nospam_test.count()}")

    val symbolTokenizer = new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("symbols")
      .setPattern("[a-zA-Z\\ ]*")

    val wordTokenizer = new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("words")
      .setPattern("[^A-Za-z]")

    val remover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filtered_words")

    val bow_symbols = new CountVectorizer()
      .setInputCol("symbols")
      .setOutputCol("raw_symbol_features")

    val bow_words = new CountVectorizer()
      .setInputCol("filtered_words")
      .setOutputCol("raw_word_features")

    val idf_symbols = new IDF()
      .setInputCol("raw_symbol_features")
      .setOutputCol("symbol_features")

    val idf_words = new IDF()
      .setInputCol("raw_word_features")
      .setOutputCol("word_features")

    val si = new StringIndexer()
      .setInputCol("label_category")
      .setOutputCol("label")

    val assembler = new VectorAssembler()
      .setInputCols(Array("symbol_features", "word_features"))
      .setOutputCol("features")

    val svm = new LinearSVC()
      .setFeaturesCol("features")
      .setLabelCol("label")

    val pipeline = new Pipeline()
      .setStages(Array(
        symbolTokenizer, wordTokenizer, remover,
        bow_symbols, bow_words,
        idf_symbols, idf_words,
        assembler, si, svm
      ))

    val multiEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setMetricName("f1")
      .setPredictionCol("prediction")

    val paramGrid = new ParamGridBuilder()
      .addGrid(bow_symbols.vocabSize, Array(1000, 2000))
      .addGrid(bow_words.vocabSize, Array(1000, 2000))
      .addGrid(svm.maxIter, Array(10, 20))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(multiEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)
      .setSeed(42)

    val cvModel = cv.fit(train_df)

    val predictionsTrain = cvModel.transform(train_df)
    val predictionsTest = cvModel.transform(test_df)

    val predictionAndLabelsTrain = predictionsTrain.select("prediction", "label").rdd.map(r => (r.getDouble(0), r.getDouble(1)))
    val predictionAndLabelsTest = predictionsTest.select("prediction", "label").rdd.map(r => (r.getDouble(0), r.getDouble(1)))

    val multiMetricsTrain = new MulticlassMetrics(predictionAndLabelsTrain)
    val multiMetricsTest = new MulticlassMetrics(predictionAndLabelsTest)

    println("---- Training ----")
    println(s"F1-Score = ${multiEvaluator.evaluate(predictionsTrain)}")
    println(s"Accuracy = ${multiMetricsTrain.accuracy}")
    println(s"Confusion matrix:\nSpam\tNotSpam\n${multiMetricsTrain.confusionMatrix}")

    println("---- Testing ----")
    println(s"F1-Score = ${multiEvaluator.evaluate(predictionsTest)}")
    println(s"Accuracy = ${multiMetricsTest.accuracy}")
    println(s"Confusion matrix:\nSpam\tNotSpam\n${multiMetricsTest.confusionMatrix}")

    spark.stop()
  }
}
