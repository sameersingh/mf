package org.sameersingh.mf.biling

import java.io.{FileInputStream, FileOutputStream, OutputStreamWriter, PrintWriter}
import java.util.zip.{GZIPOutputStream, GZIPInputStream}

import com.typesafe.scalalogging.slf4j.Logging

import scala.collection.immutable.HashMap
import scala.collection.mutable.ArrayBuffer

/**
 * @author sameer
 * @since 2/1/15.
 */
case class Pattern(pattern: String, cat: String, ners: (String, String), words: Seq[String], count: Int)

trait PatternSimilarity extends Logging {

  def similarity(en: Pattern, zh: Pattern): Double

  def process(enFile: String, zhFile: String, threshold: Double = -3.0, output: String): Unit = {
    val enPatterns = RunPatternSimilarity.readFile(enFile)
    logger.info(" # en patterns: " + enPatterns.length)
    val zhPatterns = RunPatternSimilarity.readFile(zhFile)
    logger.info(" # zh patterns: " + zhPatterns.length)
    val scores = new ArrayBuffer[(Pattern, Pattern, Double)]
    for (en <- enPatterns; if (!en.words.isEmpty);
         zh <- zhPatterns; if (!zh.words.isEmpty);
         if (RunPatternSimilarity.filter(en, zh))) {
      val sim = similarity(en, zh)
      if (sim > threshold) {
        scores += Triple(en, zh, sim)
      }
    }
    logger.info(s"Similarity computed over ${scores.size} pairs, now sorting and writing.")
    val w = new PrintWriter(new OutputStreamWriter(new FileOutputStream(output), "UTF-8"))
    for ((en, zh, sim) <- scores.sortBy(-_._3)) {
      w.println(s"${en.pattern}\t${zh.pattern}\t$sim")
    }
    w.flush()
    w.close()
  }

}

trait UsingWordSim extends PatternSimilarity {
  def wordSim: WordSimilarity
}

object RunPatternSimilarity {
  def nerMatch(nerEn: String, nerZh: String): Boolean = {
    assert(Set("PERSON", "GPE", "LOC", "ORG") contains nerZh, nerZh + " is not a supported zh NER.")
    nerEn match {
      case "PERSON" => nerEn == nerZh
      case "ORGANIZATION" => Set("ORG", "GPE").contains(nerZh)
      case "LOCATION" => Set("LOC", "GPE").contains(nerZh)
      case "DATE" => false
      case "NUMBER" => false
      case "O" => false
    }
  }

  def filter(p1: Pattern, p2: Pattern): Boolean = p1.cat == p2.cat && nerMatch(p1.ners._1, p2.ners._1) && nerMatch(p1.ners._2, p2.ners._2)

  def readFile(file: String): Seq[Pattern] = {
    val result = new ArrayBuffer[Pattern]
    val source = io.Source.fromFile(file, "UTF-8")
    for (l <- source.getLines(); if (!l.startsWith("REL$"))) {
      val split = l.split("\t")
      val pat = split(0).trim
      val cat = split(0).take(6)
      val ner1 = split(1).trim
      val ner2 = split(2).trim
      val words = split.slice(3, split.length - 1).map(_.trim).toSeq
      val count = split.last.trim.toInt
      result += Pattern(pat, cat, ner1 -> ner2, words, count)
    }
    source.close()
    result
  }

  def main(args: Array[String]): Unit = {
    val ws = new BilingEmbeddingScores()
    val ps = new MaxOverWordPairs(ws)
    val threshold = -3.0
    val output = "data/muling-re/pattern-sim.max.tsv"
    ps.process("data/muling-re/pattern.en", "data/muling-re/pattern.zh", threshold, output)
  }
}

class MaxOverWordPairs(val wordSim: WordSimilarity) extends UsingWordSim {
  override def similarity(en: Pattern, zh: Pattern): Double = {
    val scores = for (enW <- en.words; zhW <- zh.words) yield (enW, zhW, wordSim.sim(enW, zhW))
    if (scores.isEmpty) Double.NegativeInfinity
    else scores.maxBy(_._3)._3
  }
}

object FactorSimilarity extends Logging {
  def readFactors(file: String): Map[String, Array[Double]] = {
    logger.info("Reading factors")
    val result = new ArrayBuffer[(String, Array[Double])]
    val s = io.Source.fromInputStream(new GZIPInputStream(new FileInputStream(file)))("UTF-8")
    for(l <- s.getLines()) {
      val split = l.split("\\t")
      assert(split.length == 2)
      result += split(0) -> split(1).split(" ").map(_.toDouble)
    }
    s.close
    HashMap(result:_*)
  }

  def similarity(p1: String, p2: String, factors: Map[String, Array[Double]]): Double = {
    val f1 = factors(p1)
    val f2 = factors(p2)
    var sum1sq = 0.0
    var sum2sq = 0.0
    var dot = 0.0
    for((d1,d2) <- f1.zip(f2)) {
      dot += d1*d2
      sum1sq += d1*d1
      sum2sq += d2*d2
    }
    dot / (math.sqrt(sum1sq)*math.sqrt(sum2sq))
  }

  def main(args: Array[String]): Unit = {
    val dir = "data/muling-re/debug"
    val factors = readFactors(dir + "/item.factor.gz")
    val s = io.Source.fromInputStream(new GZIPInputStream(new FileInputStream(dir + "/alignSorted.txt.gz")))("UTF-8")
    val afile = new PrintWriter(new OutputStreamWriter(new GZIPOutputStream(new FileOutputStream(dir + "/patterns.sim.aligned.gz")), "UTF-8"))
    logger.info("Reading alignment pairs")
    val result = new ArrayBuffer[(String, String, Double, Double)]
    for(l <-  s.getLines) {
      val split = l.split("\t")
      assert(split.length == 3)
      val enp = split(0)
      val zhp = split(1)
      val asim = split(2).toDouble
      val fsim = similarity(enp, zhp, factors)
      val p = (enp, zhp, asim, fsim)
      afile.println(p.productIterator.mkString("\t"))
      result += p
    }
    s.close()
    afile.flush()
    afile.close()
    logger.info("Writing factor similarity")
    val ffile = new PrintWriter(new OutputStreamWriter(new GZIPOutputStream(new FileOutputStream(dir + "/patterns.sim.factors.gz")), "UTF-8"))
    result.sortBy(-_._4).foreach(p => ffile.println(p.productIterator.mkString("\t")))
    ffile.flush()
    ffile.close()
  }
}