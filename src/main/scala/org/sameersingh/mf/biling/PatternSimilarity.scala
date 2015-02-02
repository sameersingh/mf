package org.sameersingh.mf.biling

import java.io.{FileOutputStream, OutputStreamWriter, PrintWriter}

import com.typesafe.scalalogging.slf4j.Logging

import scala.collection.mutable.ArrayBuffer

/**
 * @author sameer
 * @since 2/1/15.
 */
case class Pattern(pattern: String, words: Seq[String], count: Int)

trait PatternSimilarity extends Logging {

  def similarity(en: Pattern, zh: Pattern): Double

  def process(enFile: String, zhFile: String, threshold: Double = -3.0, output: String): Unit = {
    val enPatterns = RunPatternSimilarity.readFile(enFile)
    logger.info(" # en patterns: " + enPatterns.length)
    val zhPatterns = RunPatternSimilarity.readFile(zhFile)
    logger.info(" # zh patterns: " + zhPatterns.length)
    val w = new PrintWriter(new OutputStreamWriter(new FileOutputStream(output), "UTF-8"))
    for (en <- enPatterns) {
      for (zh <- zhPatterns) {
        val sim = similarity(en, zh)
        if (sim > threshold) {
          w.println(s"${en.pattern}\t${zh.pattern}\t$sim")
        }
      }
    }
    w.flush()
    w.close()
  }

}

trait UsingWordSim extends PatternSimilarity {
  def wordSim: WordSimilarity
}

object RunPatternSimilarity {
  def readFile(file: String): Seq[Pattern] = {
    val result = new ArrayBuffer[Pattern]
    val source = io.Source.fromFile(file, "UTF-8")
    for (l <- source.getLines()) {
      val split = l.split("\t")
      val pat = split(0).trim
      val words = split.slice(1, split.length - 1).map(_.trim).toSeq
      val count = split.last.trim.toInt
      result += Pattern(pat, words, count)
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
    val scores = for(enW <- en.words; zhW <- zh.words) yield (enW, zhW, wordSim.sim(enW, zhW))
    if(scores.isEmpty) Double.NegativeInfinity
    else scores.maxBy(_._3)._3
  }
}