package org.sameersingh.mf.learner

import org.sameersingh.mf._
import cc.factorie.{optimize, Example}
import cc.factorie.util.DoubleAccumulator
import cc.factorie.la._
import cc.factorie.model.{Parameters => FParameters}
import scala.math._
import cc.factorie.optimize.{L2RegularizedConstantRate, ConstantLearningRate, OnlineTrainer}
import scala.util.Random

/**
 * @author sameer
 * @since 5/7/14.
 */
case class FactorieConfig(k: Int, lambda: Double, baseRate: Double, iterations: Int, logEvery: Int)

abstract class FactorieOptimizer(val target: ObservedMatrix, val config: FactorieConfig) {

  val random = new Random()

  def newDenseTensor2(l: Int) = {
    val t = new DenseTensor2(l, config.k)
    for (i <- 0 until t.size)
      t.update(i, random.nextDouble() / 1000.0)
    println("init: " + t.mkString(", "))
    t
  }

  class Parameters extends FParameters {
    val rowFactors = Weights(newDenseTensor2(target.rowIDs.size))
    val colFactors = Weights(newDenseTensor2(target.colIDs.size))

    def rowTensor2 = parameters(rowFactors).asInstanceOf[Tensor2]

    def colTensor2 = parameters(colFactors).asInstanceOf[Tensor2]

    def rowTensor(id: ID): Tensor1 = rowTensor2.leftMultiply(new SingletonBinaryTensor1(target.rowIDs.size, id.idx))

    def colTensor(id: ID): Tensor1 = colTensor2.leftMultiply(new SingletonBinaryTensor1(target.colIDs.size, id.idx))
  }

  class CellExample(c: Cell, params: Parameters) extends Example {
    override def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {
      val row = params.rowTensor(c.row)
      val col = params.colTensor(c.col)
      val est = pred(row, col)
      rowAndColGrads(c, row, col, est, gradient)
      value.accumulate(-loss(est, c.value.double))
    }
  }

  def rowAndColGrads(c: Cell, row: Tensor1, col: Tensor1, est: Double, gradient: WeightsMapAccumulator): Unit

  val params = new Parameters

  def avgValue(cells: Seq[Cell]): Double = cells.foldLeft(0.0)(_ + loss(_, params)) / cells.size.toDouble

  def avgValue(pred: Seq[Double], truth: Seq[Double]): Double = pred.zip(truth).foldLeft(0.0)((s, p) => s + loss(p._1, p._2)) / pred.size.toDouble

  def optimize = {
    val optimizer = new L2RegularizedConstantRate(config.lambda, config.baseRate, target.trainCells.size)
    // val trainer = new cc.factorie.optimize.BatchTrainer(params.parameters, optimizer)
    val trainer = new cc.factorie.optimize.OnlineTrainer(params.parameters, optimizer, config.iterations, config.logEvery)
    for (i <- 0 until config.iterations)
      trainer.processExamples(target.trainCells.map(c => new CellExample(c, params)))
  }

  def debug {
    println("rows: " + params.rowTensor2.mkString(", "))
    println("cols: " + params.colTensor2.mkString(", "))
  }

  def pred(row: Tensor1, col: Tensor1): Double

  def loss(pred: Double, truth: Double): Double

  def loss(c: Cell, p: Parameters): Double = {
    val row = p.rowTensor(c.row)
    val col = p.colTensor(c.col)
    loss(pred(row, col), c.value.double)
  }

}

trait LogisticLoss extends FactorieOptimizer {

  def pred(row: Tensor1, col: Tensor1): Double = {
    val score = row dot col
    Util.sigmoid(score)
  }

  def loss(pred: Double, truth: Double): Double = {
    val lprob = log(pred)
    val liprob = log(1.0 - pred)
    assert(truth >= 0.0 || truth <= 1.0)
    val l = if ((lprob.isInfinite || liprob.isInfinite) && pred == truth) 0.0 else -(truth * lprob + (1.0 - truth) * liprob) // negative log likelihood
    if (l.isNaN)
      l
    l
  }

  def rowAndColGrads(c: Cell, row: Tensor1, col: Tensor1, est: Double, gradient: WeightsMapAccumulator) {
    val rowGrads = Array.fill(row.size)(0.0)
    val colGrads = Array.fill(col.size)(0.0)
    val direction = (c.value.double - est) // gradient of negative log-likelihood
    // do the rows first
    for (k <- 0 until row.size) {
      rowGrads(k) = direction * col(k)
    }
    gradient.accumulate(params.rowFactors, new SingletonLayeredTensor2(target.rowIDs.size, config.k, c.row.idx, 1.0, new DenseTensor1(rowGrads)))
    // then do the cols
    for (k <- 0 until col.size) {
      colGrads(k) = direction * row(k)
    }
    gradient.accumulate(params.colFactors, new SingletonLayeredTensor2(target.colIDs.size, config.k, c.col.idx, 1.0, new DenseTensor1(colGrads)))
  }
}

trait L2Loss extends FactorieOptimizer {
  override def loss(pred: Double, truth: Double): Double = StrictMath.pow(truth - pred, 2.0)

  override def pred(row: Tensor1, col: Tensor1): Double = row dot col

  override def rowAndColGrads(c: Cell, row: Tensor1, col: Tensor1, est: Double, gradient: WeightsMapAccumulator): Unit = {
    val rowGrads = Array.fill(row.size)(0.0)
    val colGrads = Array.fill(col.size)(0.0)
    val err = c.value.double - est
    // do the rows first
    for (k <- 0 until row.size) {
      rowGrads(k) = 2.0 * err * col(k)
    }
    for (k <- 0 until col.size) {
      colGrads(k) = 2.0 * err * row(k)
    }
    gradient.accumulate(params.rowFactors, new SingletonLayeredTensor2(target.rowIDs.size, config.k, c.row.idx, 1.0, new DenseTensor1(rowGrads)))
    gradient.accumulate(params.colFactors, new SingletonLayeredTensor2(target.colIDs.size, config.k, c.col.idx, 1.0, new DenseTensor1(colGrads)))
  }
}
