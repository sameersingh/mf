package org.sameersingh.mf

import scala.math._

trait PredictValue {
  def predictParams: Seq[Parameters]

  def target: ObservedMatrix

  def pred(c: Cell): Double
}

trait PredictProb extends PredictValue {

  import math._

  def prob(c: Cell): Double = min(1.0, max(0.0, pred(c)))

  def logProbs(c: Cell): (Double, Double)
}

class DotValue(val rowFactors: DoubleDenseMatrix,
               val colFactors: DoubleDenseMatrix,
               val target: ObservedMatrix) extends PredictValue {

  def this(params: ParameterSet, u: String, v: String, target: ObservedMatrix) =
    this(params(u), params(v), target)

  assert(rowFactors.numCols == colFactors.numCols, "Number of columns for DotTerms should match %s->%d, %s->%d" format(rowFactors.name, rowFactors.numCols, colFactors.name, colFactors.numCols))

  private val _params: Seq[Parameters] = Seq(rowFactors, colFactors)

  def predictParams = _params

  def dot(c: Cell) = rowFactors.r(c.row).zip(colFactors.r(c.col)).foldLeft(0.0)((s, uv) => s + uv._1 * uv._2)

  def pred(c: Cell): Double = dot(c)
}

class DotValueWithBias(rowFactors: DoubleDenseMatrix,
                       colFactors: DoubleDenseMatrix,
                       val rowBias: ParamVector,
                       val colBias: ParamVector,
                       target: ObservedMatrix) extends DotValue(rowFactors, colFactors, target) {
  def this(params: ParameterSet, u: String, v: String, target: ObservedMatrix) =
    this(params(u), params(v), params.f(u, "bias"), params.f(v, "bias"), target)

  private val _params: Seq[Parameters] = super.predictParams ++ Seq(rowBias, colBias)

  override def predictParams: Seq[Parameters] = _params

  override def dot(c: Cell): Double = super.dot(c) + rowBias(c.row) + colBias(c.col)
}

trait LogisticDot extends DotValue with PredictProb {
  override def pred(c: Cell): Double = {
    val score = dot(c)
    val escore = exp(score)
    escore / (escore + 1.0)
  }

  def logProbs(c: Cell) = {
    val score = dot(c)
    val logZ = log(exp(score) + 1.0)
    val lprob = score - logZ
    val liprob = -logZ // log(1) - logZ
    (lprob, liprob)
  }
}

/**
 * A single term in the objective function, defining the value and gradient.
 * The objective is minimized, and therefore value and gradient should be computed assuming minimization.
 */
trait Term {
  def weight: ParamDouble

  def params: Seq[Parameters]

  def gradient(c: Cell): Gradients
}

trait ErrorTerm extends Term with Error {
  def params: Seq[Parameters] = predictValue.predictParams
}

/**
 * A term that L2 distance between the true value and the dot product of the row and column factors for the given matrix
 */
class DotL2(val predictValue: DotValue,
            val weight: ParamDouble) extends L2 with ErrorTerm {
  def this(params: ParameterSet, dot: DotValue, w: Double) =
    this(dot, params(dot.target, "weight", w))

  def this(params: ParameterSet, u: String, v: String, w: Double, target: ObservedMatrix) =
    this(params, new DotValue(params, u, v, target), w)

  def gradient(c: Cell): Gradients = {
    val grads = new Gradients
    val row = predictValue.rowFactors.r(c.row)
    val col = predictValue.colFactors.r(c.col)
    val rowGrads = Array.fill(predictValue.rowFactors.numCols)(0.0)
    val colGrads = Array.fill(predictValue.colFactors.numCols)(0.0)
    // compute the error for the cell
    val err = c.value.double - predictValue.pred(c)
    // do the rows first
    for (k <- 0 until predictValue.rowFactors.numCols) {
      rowGrads(k) = -2.0 * weight() * err * col(k)
    }
    grads(predictValue.rowFactors) = (c.row -> rowGrads)
    // then do the cols
    for (k <- 0 until predictValue.colFactors.numCols) {
      colGrads(k) = -2.0 * weight() * err * row(k)
    }
    grads(predictValue.colFactors) = (c.col -> colGrads)
    grads
  }
}

/**
 * Introduces biases for rows and columns, but otherwise identical to L2DotTerm
 */
class DotWithBiasL2(predictValue: DotValueWithBias, weight: ParamDouble) extends DotL2(predictValue, weight) {
  def this(params: ParameterSet, dot: DotValueWithBias, w: Double) =
    this(dot, params(dot.target, "weight", w))

  def this(params: ParameterSet, u: String, v: String, w: Double, target: ObservedMatrix) =
    this(params, new DotValueWithBias(params, u, v, target), w)

  override def gradient(c: Cell): Gradients = {
    val grads = super.gradient(c)
    val err = c.value.double - predictValue.pred(c)
    val g = -2.0 * weight() * err
    grads(predictValue.rowBias) = (c.row -> Array(g))
    grads(predictValue.colBias) = (c.col -> Array(g))
    grads
  }
}

/**
 * Logistic loss suitable for binary matrices. The term is optimized by minimizing negative log-likelihood.
 */
class LogisticDotNLL(val predictValue: DotValue with LogisticDot,
                     val weight: ParamDouble) extends NLL with ErrorTerm {
  def this(params: ParameterSet, dot: DotValue with LogisticDot, w: Double) =
    this(dot, params(dot.target, "weight", w))

  def gradient(c: Cell): Gradients = {
    val grads = new Gradients
    val row = predictValue.rowFactors.r(c.row)
    val col = predictValue.colFactors.r(c.col)
    val rowGrads = Array.fill(predictValue.rowFactors.numCols)(0.0)
    val colGrads = Array.fill(predictValue.colFactors.numCols)(0.0)
    val prob = predictValue.pred(c)
    val direction = -(c.value.double - prob) // gradient of negative log-likelihood
    // do the rows first
    for (k <- 0 until predictValue.rowFactors.numCols) {
      rowGrads(k) = weight() * direction * col(k)
    }
    grads(predictValue.rowFactors) = (c.row -> rowGrads)
    // then do the cols
    for (k <- 0 until predictValue.colFactors.numCols) {
      colGrads(k) = weight() * direction * row(k)
    }
    grads(predictValue.colFactors) = (c.col -> colGrads)
    grads
  }
}

/**
 * Include bias in the dot term, but otherwise same as LogisticDotTerm
 */
class LogisticDotWithBiasNLL(predictValue: DotValueWithBias with LogisticDot,
                             weight: ParamDouble) extends LogisticDotNLL(predictValue, weight) {
  override def gradient(c: Cell): Gradients = {
    val grads = super.gradient(c)
    val prob = predictValue.prob(c)
    val direction = -(c.value.double - prob) // gradient of negative log-likelihood
    val g = weight() * direction
    grads(predictValue.rowBias) = (c.row -> Array(g))
    grads(predictValue.colBias) = (c.col -> Array(g))
    grads
  }
}

/**
 * L2 Regularization for a given factor
 */
class L2Regularization(val factors: DoubleDenseMatrix, val weight: ParamDouble, val numCells: Int = 1)
  extends Term {
  def this(params: ParameterSet, f: String, n: Int) = this(params(f), params.l2RegCoeff(f), n)

  val params: Seq[Parameters] = Seq(factors)

  def value(id: ID): Double = overallWeight * factors.r(id).map(u => u * u).sum

  def value: Double = factors.rids.map(id => value(id)).sum

  // for stochastic estimation, the value for a cell
  def value(c: Cell): Double =
    if (c.row.domain == factors.name) {
      value(c.row) / numCells
    } else if (c.col.domain == factors.name) {
      value(c.col) / numCells
    } else 0.0

  lazy val overallWeight = weight() * (1.0 / (factors.numCols * factors.rids.size))
  lazy val gradientWeight = 2.0 * overallWeight / numCells

  def gradient(c: Cell): Gradients = {
    val grads = new Gradients
    if (c.row.domain == factors.name) {
      val gs = Array.fill(factors.numCols)(0.0)
      val row = factors.r(c.row)
      for (k <- 0 until factors.numCols) {
        gs(k) = gradientWeight * row(k)
      }
      grads(factors) = (c.row -> gs)
    } else if (c.col.domain == factors.name) {
      val gs = Array.fill(factors.numCols)(0.0)
      val col = factors.r(c.col)
      for (k <- 0 until factors.numCols) {
        gs(k) = gradientWeight * col(k)
      }
      grads(factors) = (c.col -> gs)
    }
    grads
  }
}