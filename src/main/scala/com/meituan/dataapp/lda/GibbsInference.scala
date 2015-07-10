package com.meituan.dataapp.lda

import org.apache.spark.broadcast.Broadcast


class GibbsInference(val iters: Int)
  extends Serializable{

  def inference(alpha: Double, phiBC: Broadcast[Array[Array[Double]]], doc: Array[Int]) : Array[Double] = {
    val phi = phiBC.value
    val K = phi.length
    val nd = Array.ofDim[Int](K)
    var ndsum = 0

    val N = doc.length
    val z = Array.ofDim[Int](N)
    for (n <- Range(0, N)) {
      val topic = (Math.random() * K).toInt
      z(n) = topic
      nd(topic) += 1
    }

    ndsum = N
    for (i <- Range(0, iters)) {
      for (n <- Range(0, N)) {
        var topic = z(n)
        nd(topic) -= 1
        ndsum -= 1

        val p = Array.ofDim[Double](K)
        for (k <- Range(0, K)) {
          p(k) = (nd(k) + alpha) / (ndsum + K * alpha) * phi(k)(doc(n))
        }

        for (k <- Range(1, K)) {
          p(k) += p(k - 1)
        }

        val u = Math.random() * p(K - 1)
        var flag = true
        for (i <- Range(0, K)) {
          if (flag && u < p(i)) {
            topic = i
            flag = false
          }
        }

        nd(topic) += 1
        ndsum += 1
        z(n) = topic
      }
    }
    val theta = Array.ofDim[Double](K)
    for (k <- Range(0, K)) {
      theta(k) = (nd(k) + alpha) / (ndsum + K * alpha)
    }

    theta
  }

}
