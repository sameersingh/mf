package org.sameersingh.mf.biling

import scala.collection.mutable

/**
 * Created by sameer on 1/30/15.
 */
class BilingEmbeddingScores(bilingMTDir: String = "data/muling-re/biling") extends WordSimilarity {

  val zhEmbds = new mutable.HashMap[String, Array[Double]] {
    override def default(key: String): Array[Double] = Array.empty
  }
  val enEmbds = new mutable.HashMap[String, Array[Double]] {
    override def default(key: String): Array[Double] = Array.empty
  }

  init(bilingMTDir)

  def readFile(filename: String, map: mutable.HashMap[String, Array[Double]]): Unit = {
    logger.info("Reading " + filename)
    logger.info("# words: " + map.size)
    val source = io.Source.fromFile(filename)
    for(line <- source.getLines()) {
      val split = line.split("\\s")
      assert(split.length == 2)
      map(split(0)) = split(1).split(",").map(_.toDouble)
    }
    source.close()
    logger.info("Done reading " + filename)
    logger.info("# words: " + map.size)
  }

  def init(bilingMTDir: String) = {
    enEmbds.clear
    readFile(bilingMTDir + "/En_vectors.txt", enEmbds)
    zhEmbds.clear
    readFile(bilingMTDir + "/biling_trained_Zh_vectors.txt", zhEmbds)
  }

  def sim(arr1: Array[Double], arr2: Array[Double]): Double = {
    if(arr1.isEmpty || arr2.isEmpty) Double.NegativeInfinity
    else {
      assert(arr1.size == arr2.size)
      // arr1.zip(arr2).map(p => p._1 * p._2).sum
      -math.sqrt(arr1.zip(arr2).map(p => math.pow(p._1 - p._2, 2)).sum)
    }
  }

  override def sim(en: String, zh: String): Double = sim(enEmbds(en),zhEmbds(zh))

  override def enMono(w1: String, w2: String): Double = sim(enEmbds(w1),enEmbds(w2))

  override def zhMono(w1: String, w2: String): Double = sim(zhEmbds(w1),zhEmbds(w2))

  def nearest(arr: Array[Double],
              map: Iterable[(String, Array[Double])],
              k: Int = 1): Seq[(String, Double)] = {
    map.map(we => we._1 -> sim(we._2,arr)).toSeq.sortBy(-_._2).take(k)
  }

  override def nearestEn(zh: String, k: Int = 1) = nearest(zhEmbds(zh), enEmbds, k)

  override def nearestZh(en: String, k: Int = 1) = nearest(enEmbds(en), zhEmbds, k)

  override def nearestEnMono(en: String, k: Int = 1) = nearest(enEmbds(en), enEmbds, k)

  override def nearestZhMono(zh: String, k: Int = 1) = nearest(zhEmbds(zh), zhEmbds, k)
}

object BilingEmbeddingScores {
  def main(args: Array[String]): Unit = {
    val dir = "data/muling-re/biling"
    val sim = new BilingEmbeddingScores(dir)
    val words = Seq("born", "wife", "married", "president")
    for(w <- words) {
      println("Word: " + w)
      println("-- English")
      for(en <- sim.nearestEnMono(w, 10))
        println("      " + en._1 + "\t" + en._2)
      println("-- Chinese")
      for(zh <- sim.nearestZh(w, 10))
        println("      " + zh._1 + "\t" + zh._2)
    }
  }
}
