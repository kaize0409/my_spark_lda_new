package com.meituan.dataapp.lda

import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.feature.Normalizer
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.util.Vector
import org.apache.spark.{SparkConf, SparkContext}

object Train {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("LDA Train Application")
    val sc = new SparkContext(conf)

    val numberOfTopics = args(0).toInt
    val numberOfIterations = args(1).toInt
    val doc_input_path = args(2)
    val base_output_path = args(3)
    val numOfPartitions = args(4).toInt
    val top_topic_output_path = base_output_path + "/top_phi"
    val topic_output_path = base_output_path + "/phi"
    val doc_output_path = base_output_path + "/theta"
    val vocab_output_path = base_output_path + "/vocab"

    // Load and parse the data
    val rawDocuments = sc.textFile(doc_input_path).map(_.split("\t")).filter(_.size >= 2)
      .map(x => (x(0), x(1).split(":").toSeq))
    val tokenIndexer = new TokenEnumerator().setRareTokenThreshold(100)
    val tokenIndex = tokenIndexer(rawDocuments)
    val tokenIndexBC = sc.broadcast(tokenIndex)

    val data = rawDocuments.map(tokenIndexBC.value.transform)
    val docNameArray = data.map(_.docName).collect()
    val parsedData = data.repartition(numOfPartitions).map(s => Vectors.dense(s.tokens.map(_.toDouble)))
    // Index documents with unique IDs
    val corpus = parsedData.zipWithIndex.map(_.swap).cache()

    println("--------phase1: Save vocab--------")
    val vocab = tokenIndex.alphabet.toArray.mkString(":")
    sc.parallelize(Array(vocab)).saveAsTextFile(vocab_output_path)

    println("--------phase2: Train model--------")
    // Cluster the documents into three topics using LDA
    val lda = new LDA()
    val ldaModel = lda.setK(numberOfTopics).setMaxIterations(numberOfIterations).run(corpus)

    println("--------phase3: Save topic's top words--------")
    val topWordCount = 30
    val topWords = ldaModel.describeTopics(topWordCount).map {
      x => {
        val str = new StringBuilder
        for (i <- Range(0, topWordCount)) {
          if (i != 0)
            str.append(";")
          str.append(tokenIndexBC.value.alphabet.get(x._1(i)) + ":" + x._2(i).formatted("%.4f"))
        }
        str.toString()
      }
    }
    sc.parallelize(topWords).saveAsTextFile(top_topic_output_path)

    println("--------phase4: Save topic's distribution over words--------")
    val phi = ldaModel.describeTopics().map {
      x => {
        val str = new StringBuilder
        for (i <- Range(0, x._1.length)) {
          if (i != 0)
            str.append(";")
          str.append(x._1(i) + ":" + x._2(i))
        }
        str.toString()
      }
    }
    sc.parallelize(phi).saveAsTextFile(topic_output_path)

    println("--------phase5: Save document's distribution over topics-------")
    val doc_topic = ldaModel.topicDistributions
    val theta = doc_topic.map {
      case (docID, theta) => {
        docNameArray(docID.toInt) + "\t" + theta.toArray.zipWithIndex.sortBy(-_._1).map(x => x._2 + ":" + x._1).mkString(";")
      }
    }
    theta.saveAsTextFile(doc_output_path)

    sc.stop()
  }

  def cos_sim(theta1: Array[Double], theta2: Array[Double]): Double = {
    val va = Vectors.dense(theta1)
    val vb = Vectors.dense(theta2)
    val normalizer = new Normalizer()
    // normalizer.transform(va) dot normalizer.transform(vb)
    val norm_a = Vector(normalizer.transform(va).toArray)
    val norm_b = Vector(normalizer.transform(vb).toArray)
    norm_a.dot(norm_b)

  }

  def js_sim(theta1: Array[Double], theta2: Array[Double]): Double = {
    var sum1 = 0.0
    for (i <- Range(0, theta1.length)) {
      sum1 = sum1 + theta1(i) * math.log(theta1(i) / theta2(i))
    }
    var sum2 = 0.0
    for (i <- Range(0, theta1.length)) {
      sum2 = sum2 + theta2(i) * math.log(theta2(i) / theta1(i))
    }
    (sum1 + sum2) / 2
  }

}
