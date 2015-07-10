package com.meituan.dataapp.lda

import breeze.util.Index
import org.apache.spark.{SparkContext, SparkConf}

object Inference{
  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("LDA GibbsInference Application")
    val sc = new SparkContext(conf)

    // 迭代次数
    val numberOfIterations = args(0).toInt
    // alpha超参数
    val alpha = args(1).toDouble
    // 词表路径
    val vocab_input_path = args(2)
    // phi矩阵路径
    val phi_path = args(3)
    // 需要inference的Doc路径
    val query_input_path = args(4)
    // inference结果保存路径
    val query_output_path = args(5)

    val numOfPartitions = args(6).toInt

    val vocab = sc.textFile(vocab_input_path).first().split(":")
    val tokenIndex = new TokenEnumeration(Index(vocab))
    val gibbs = new GibbsInference(numberOfIterations)
    val queries = sc.textFile(query_input_path)
      .map(_.split("\t")).filter(_.size >= 2)
      .map(x => (x(0), x(1).split(":")))
      .map(q => (q._1, q._2.map(tokenIndex.alphabet.apply).filter(p => p>0)))

    val vocabSize = tokenIndex.alphabet.size
    val phi = sc.textFile(phi_path).map {
      line => {
        val weights = Array.ofDim[Double](vocabSize)
        val pairs = line.split(";")

        pairs.foreach(pair =>{
          val items = pair.split(":")
          weights(items(0).toInt) = items(1).toDouble
        })
        weights
      }
    }
    val phiBC = sc.broadcast(phi.collect())

    val thetas = queries.repartition(numOfPartitions).map {
      d => (d._1, gibbs.inference(alpha, phiBC, d._2))
    }
    val query_topic_result = thetas.map {
      theta => {
        theta._1 + "\t" + theta._2.zipWithIndex.sortBy(-_._1).map(x => x._2 + ":" + x._1).mkString(";")
      }
    }
    query_topic_result.saveAsTextFile(query_output_path)

    sc.stop()

  }
}

