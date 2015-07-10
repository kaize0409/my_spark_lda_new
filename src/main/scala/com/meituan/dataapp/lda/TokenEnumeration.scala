package com.meituan.dataapp.lda

import breeze.util.Index
import org.apache.spark.rdd.RDD

class Document(val docName: String, val tokens: Array[Int]) extends Serializable {
  /**
   *
   * @return number of different tokens in the collection
   */
  def alphabetSize = tokens.length
}

class TokenEnumeration(val alphabet: Index[String]) extends Serializable {

  /**
   *
   * @return a Document that contains all the tokens
   *         from rawDocument that are included in the alphabet
   */

  def transform(rawDocument: (String, Seq[String])): Document = {
    val wordsMap = rawDocument._2.map(alphabet.apply)
      .filter(_ != -1)
      .foldLeft(Map[Int, Int]().withDefaultValue(0))((map, word) => map + (word -> (1 + map(word))))

    val words = wordsMap.keys.toArray.sorted

    val tokens = Array.fill[Int](alphabet.size)(0)
    words.map {
      index => {
        tokens(index) = wordsMap(index)
      }
    }
    /*val words = wordsMap.keys.toArray.sorted
    val tokens = new SparseVector[Int](words, words.map(word => wordsMap(word)), alphabet.size)//返回词的index和出现次数
    // new Document(tokens)
    new Document(rawDocument._1, tokens)

    val tokens = rawDocument._2.map(alphabet.apply).filter(_ != -1)*/
    new Document(rawDocument._1, tokens)
  }
}

class TokenEnumerator extends Serializable {
  private var rareTokenThreshold: Int = 2

  /**
   * @param rareTokenThreshold tokens that are encountered in the collection less than
   *                           rareTokenThreshold times are omitted.
   *                           Default value: 2
   */
  def setRareTokenThreshold(rareTokenThreshold: Int) = {
    this.rareTokenThreshold = rareTokenThreshold
    this
  }

  /**
   *
   * @param rawDocuments RDD of tokenized documents (every document is a sequence of tokens
   *                     (Strings) )
   * @return a TokenEnumeration
   */

  def apply(rawDocuments: RDD[(String, Seq[String])]): TokenEnumeration = {
    val alphabet = Index(rawDocuments.flatMap(x => x._2)
      .map((_, 1))
      .reduceByKey(_ + _)
      .filter(_._2 > rareTokenThreshold)
      .map(_._1)
      .collect)
    new TokenEnumeration(alphabet)
  }
}

