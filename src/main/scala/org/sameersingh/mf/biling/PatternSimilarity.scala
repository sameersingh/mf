package org.sameersingh.mf.biling

import java.io.{FileOutputStream, OutputStreamWriter, PrintWriter}

import com.typesafe.scalalogging.slf4j.Logging

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
    val threshold = 0.05
    val output = "data/muling-re/pattern-sim.max.tsv"
    ps.process("data/muling-re/pattern.en", "data/muling-re/pattern.zh", threshold, output)
  }
}

class MaxOverWordPairs(val wordSim: WordSimilarity) extends UsingWordSim {
  override def similarity(en: Pattern, zh: Pattern): Double = {
    val scores = for (enW <- en.words; zhW <- zh.words) yield (enW, zhW, wordSim.sim(enW, zhW))
    if (scores.isEmpty) 0.0
    else math.exp(scores.maxBy(_._3)._3)
  }
}