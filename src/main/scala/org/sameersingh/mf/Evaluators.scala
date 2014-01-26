package org.sameersingh.mf

import cc.factorie.util.coref.CorefEvaluator.Metric

/**
 * @author sameer
 */
trait Evaluator {
  def eval(cells: Seq[Cell], prefix: String): Map[String, Double]

  def evalTrain(m: ObservedMatrix) = eval(m.trainCells, "Train ")

  def evalTest(m: ObservedMatrix) = eval(m.testCells, "Test ")
}

trait Error extends Evaluator {
  def predictValue: PredictValue

  // for stochastic estimation, the value for a cell
  def value(c: Cell): Double

  def avgValue(cells: Seq[Cell]): Double = cells.foldLeft(0.0)(_ + value(_)) / cells.size.toDouble

  def name: String

  def eval(cells: Seq[Cell], prefix: String): Map[String, Double] = Map(prefix + name -> avgValue(cells))
}

trait L2 extends Error {
  val name = "L2"

  def value(c: Cell): Double = if (c.inMatrix == predictValue.target) {
    StrictMath.pow(c.value.double - predictValue.pred(c), 2.0)
  } else 0.0
}

trait Hamming extends Error {
  val name = "Hamming"

  def predictValue: PredictProb

  def threshold = 0.5

  def value(c: Cell): Double = if (c.inMatrix == predictValue.target) {
    val prob = predictValue.prob(c)
    if ((prob > threshold && c.value.double > 0.5) || (prob <= threshold && c.value.double <= 0.5))
      0.0
    else
      100.0
  } else 0.0
}

trait NLL extends Error {
  val name = "NLL"

  def predictValue: PredictProb

  def value(c: Cell): Double = if (c.inMatrix == predictValue.target) {
    // log prob of c.value
    val (lprob, liprob) = predictValue.logProbs(c)
    assert(c.value.double >= 0.0 || c.value.double <= 1.0)
    -(c.value.double * lprob + (1.0 - c.value.double) * liprob) // negative log likelihood
  } else 0.0
}

trait PerCellF1 extends Evaluator {
  def predictValue: PredictProb

  def threshold = 0.5

  def eval(c: Cell) = if (c.inMatrix == predictValue.target) {
    val prob = predictValue.prob(c)
    if (prob > threshold) {
      if (c.value.double > 0.5) tp += 1.0
      else fp += 1.0
    } else {
      if (c.value.double > 0.5) fn += 1.0
    }
  }

  var tp: Double = 0.0
  var fp: Double = 0.0
  var fn: Double = 0.0

  def precNumerator: Double = tp

  def precDenominator: Double = tp + fp

  def recallNumerator: Double = tp

  def recallDenominator: Double = tp + fn

  def precision: Double = {
    if (precDenominator == 0.0) {
      1.0
    } else {
      precNumerator / precDenominator
    }
  }

  def recall: Double = {
    if (recallDenominator == 0.0) {
      1.0
    } else {
      recallNumerator / recallDenominator
    }
  }

  def f1: Double = {
    val r: Double = recall
    val p: Double = precision
    if (p + r == 0.0) 0.0
    else (2 * p * r) / (p + r)
  }

  def reset = {
    tp = 0.0
    fp = 0.0
    fn = 0.0
  }

  def eval(cells: Seq[Cell], prefix: String) = {
    reset
    for (c <- cells) eval(c)
    Map("Prec" -> precision, "Recall" -> recall, "F1" -> f1).map(p => (prefix + p._1, p._2 * 100.0))
  }
}

class Evaluators(val evals: Seq[Evaluator]) {
  def string(m: ObservedMatrix, additionalTestCells: Seq[Cell]) = {
    val combinedTest = m.testCells ++ additionalTestCells
    (for (e <- evals) yield {
      val train = e.evalTrain(m)
      val test = e.evalTest(m)
      val additional = e.eval(additionalTestCells, "Additional ")
      val combined = e.eval(combinedTest, "Combined Test ")
      for (r <- train ++ test ++ additional ++ combined) yield (r._1 + "\t:\t" + r._2)
    }).flatten.mkString("\n")
  }
}

object Evaluators {
  def apply(evals: Evaluator*) = new Evaluators(evals)
}
