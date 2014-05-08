package org.sameersingh.mf

import StrictMath._

/**
 * @author sameer
 * @since 5/8/14.
 */
object Util {

  def sigmoid(score: Double) = {
    if (score <= 0.0) {
      val escore = exp(score)
      escore / (escore + 1.0)
    } else {
      val escore = exp(-score)
      1.0 / (escore + 1.0)
    }
  }

}
