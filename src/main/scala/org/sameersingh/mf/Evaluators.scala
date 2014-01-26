package org.sameersingh.mf

import math._
import scala.collection.mutable

/**
 * @author sameer
 */
trait Evaluator {
  def predictValue: PredictValue

  def evaluate(predTruths: Seq[(Double, Double)], prefix: String = ""): Seq[(String, Double)] = eval(predTruths.map(_._1), predTruths.map(_._2), prefix)

  def eval(pred: Seq[Double], truth: Seq[Double], prefix: String): Seq[(String, Double)]

  def eval(cells: Seq[Cell], prefix: String): Seq[(String, Double)] = eval(cells.map(c => predictValue.pred(c)), cells.map(_.value.double), prefix)

  def evalTrain(m: ObservedMatrix) = eval(m.trainCells, "Train ")

  def evalTest(m: ObservedMatrix) = eval(m.testCells, "Test ")
}

trait Error extends Evaluator {
  // for stochastic estimation, the value for a cell
  def value(c: Cell): Double = if (c.inMatrix == predictValue.target) {
    value(predictValue.pred(c), c.value.double)
  } else 0.0

  def value(pred: Double, truth: Double): Double

  def avgValue(cells: Seq[Cell]): Double = cells.foldLeft(0.0)(_ + value(_)) / cells.size.toDouble

  def avgValue(pred: Seq[Double], truth: Seq[Double]): Double = pred.zip(truth).foldLeft(0.0)((s, p) => s + value(p._1, p._2)) / pred.size.toDouble

  def name: String

  def eval(pred: Seq[Double], truth: Seq[Double], prefix: String) = Seq(prefix + name -> avgValue(pred, truth))
}

trait L2 extends Error {
  val name = "L2"

  def value(pred: Double, truth: Double): Double = StrictMath.pow(truth - pred, 2.0)
}

trait Hamming extends Error {
  val name = "Hamming"

  def predictValue: PredictProb

  def threshold = 0.5

  def value(pred: Double, truth: Double): Double = {
    val prob = pred
    if ((prob > threshold && truth > 0.5) || (prob <= threshold && truth <= 0.5))
      0.0
    else
      100.0
  }
}

trait NLL extends Error {
  val name = "NLL"

  def predictValue: PredictProb

  // log prob of c.value
  def value(pred: Double, truth: Double): Double = {
    val lprob = log(pred)
    val liprob = log(1.0 - pred)
    assert(truth >= 0.0 || truth <= 1.0)
    -(truth * lprob + (1.0 - truth) * liprob) // negative log likelihood
  }
}

trait PerCellF1 extends Evaluator {
  def predictValue: PredictProb

  def threshold = 0.5

  def eval(pred: Double, truth: Double) = {
    if (pred > threshold) {
      if (truth > 0.5) tp += 1.0
      else fp += 1.0
    } else {
      if (truth > 0.5) fn += 1.0
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

  def eval(pred: Seq[Double], truth: Seq[Double], prefix: String): Seq[(String, Double)] = {
    reset
    for (p <- pred.zip(truth)) eval(p._1, p._2)
    Seq("Prec" -> precision, "Recall" -> recall, "F1" -> f1).map(p => (prefix + p._1, p._2 * 100.0))
  }

}

class Evaluators(val evals: Seq[Evaluator]) {

  val dotValues = evals.groupBy(_.predictValue)

  def eval(m: ObservedMatrix, additionalTestCells: Seq[Cell]): Map[String, Seq[(String, Double)]] = {
    val result = new mutable.HashMap[String, Seq[(String, Double)]]
    // prepare data
    val combinedTest = m.testCells ++ additionalTestCells
    // go through the dot values
    for (dv <- dotValues.keys; if (dv.target == m)) {
      val additionalCells = additionalTestCells.filter(_.inMatrix == m)
      // predict values for each cell only once
      val trainPreds = "Train " -> m.trainCells.map(c => (dv.pred(c), c.value.double))
      val testPred = "Test " -> m.testCells.map(c => (dv.pred(c), c.value.double))
      val additionalPreds = "Additional " -> additionalCells.map(c => (dv.pred(c), c.value.double))
      val combinedPreds = "Combined " -> (testPred._2 ++ additionalPreds._2)
      val evalPreds = Seq(trainPreds, testPred, additionalPreds, combinedPreds)
      for (evalPred <- evalPreds) {
        result(evalPred._1) = evals.map(e => e.evaluate(evalPred._2)).flatten
      }
    }
    result.toMap
  }

  def string(evalResults: Map[String, Seq[(String, Double)]]): String = {
    val sb = new StringBuffer
    sb append ("%14s %s\n" format("Data type", evalResults.head._2.map(_._1).map(s => "%13s" format (s)).mkString(" ")))
    sb append ("%s %s\n" format(("%14s" format("")).replace(' ', '-'), evalResults.head._2.map(_._1).map(s => "%13s" format ("")).map(_.replace(' ', '-')).mkString(" ")))
    for ((data, results) <- evalResults) {
      sb append ("%15s%s\n" format(data, results.map(_._2).map(s => "%13.7f" format (s)).mkString(" ")))
    }
    sb append ("%s %s\n" format(("%14s" format("")).replace(' ', '-'), evalResults.head._2.map(_._1).map(s => "%13s" format ("")).map(_.replace(' ', '-')).mkString(" ")))
    sb.toString
  }

  def string(m: ObservedMatrix, additionalTestCells: Seq[Cell]): String = string(eval(m, additionalTestCells))
}

object Evaluators {
  def apply(evals: Evaluator*) = new Evaluators(evals)
}
