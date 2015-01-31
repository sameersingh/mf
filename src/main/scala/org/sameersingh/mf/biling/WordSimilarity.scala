package org.sameersingh.mf.biling

import com.typesafe.scalalogging.slf4j.Logging

/**
 * Created by sameer on 1/30/15.
 */
trait WordSimilarity extends Logging {

  def sim(en: String, zh: String): Double

  def enMono(w1: String, w2: String): Double

  def zhMono(w1: String, w2: String): Double

  def nearestEn(zh: String, k: Int = 1): Seq[(String, Double)]

  def nearestZh(en: String, k: Int = 1): Seq[(String, Double)]

  def nearestEnMono(en: String, k: Int = 1): Seq[(String, Double)]

  def nearestZhMono(zh: String, k: Int = 1): Seq[(String, Double)]
}
