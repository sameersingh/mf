package org.sameersingh.mf

import scala.math._

/**
 * A single term in the objective function, defining the value and gradient.
 * The objective is minimized, and therefore value and gradient should be computed assuming minimization.
 */
trait Term {
  def params: Seq[Parameters]

  // for stochastic estimation, the value for a cell
  def value(c: Cell): Double

  def gradient(c: Cell): Gradients

  def avgTrainingValue(m: ObservedMatrix): Double = m.trainCells.foldLeft(0.0)(_ + value(_)) / m.trainCells.size.toDouble

  def avgTestValue(m: ObservedMatrix): Double = m.testCells.foldLeft(0.0)(_ + value(_)) / m.testCells.size.toDouble
}

abstract class DotTerm(val rowFactors: DoubleDenseMatrix,
                       val colFactors: DoubleDenseMatrix,
                       val weight: ParamDouble,
                       val target: ObservedMatrix) extends Term {

  def this(params: ParameterSet, u: String, v: String, w: Double, target: ObservedMatrix) =
    this(params(u), params(v), params(target, "weight", w), target)

  assert(rowFactors.numCols == colFactors.numCols, "Number of columns for DotTerms should match %s->%d, %s->%d" format(rowFactors.name, rowFactors.numCols, colFactors.name, colFactors.numCols))

  private val _params: Seq[Parameters] = Seq(rowFactors, colFactors)

  def params = _params

  def dot(c: Cell) = rowFactors.r(c.row).zip(colFactors.r(c.col)).foldLeft(0.0)((s, uv) => s + uv._1 * uv._2)
}

abstract class DotTermWithBias(rowFactors: DoubleDenseMatrix,
                               colFactors: DoubleDenseMatrix,
                               val rowBias: ParamVector,
                               val colBias: ParamVector,
                               weight: ParamDouble,
                               target: ObservedMatrix) extends DotTerm(rowFactors, colFactors, weight, target) {
  def this(params: ParameterSet, u: String, v: String, w: Double, target: ObservedMatrix) =
    this(params(u), params(v), params.f(u, "bias"), params.f(v, "bias"), params(target, "weight", w), target)

  private val _params: Seq[Parameters] = super.params ++ Seq(rowBias, colBias)

  override def params: Seq[Parameters] = _params

  override def dot(c: Cell): Double = super.dot(c) + rowBias(c.row) + colBias(c.col)
}


/**
 * A term that L2 distance between the true value and the dot product of the row and column factors for the given matrix
 */
trait L2 extends DotTerm {

  def error(c: Cell) = c.value.double - dot(c)

  // for stochastic estimation, the value for a cell
  def value(c: Cell): Double = if (c.inMatrix == target) {
    StrictMath.pow(error(c), 2.0)
  } else 0.0

  def gradient(c: Cell): Gradients = {
    val grads = new Gradients
    val row = rowFactors.r(c.row)
    val col = colFactors.r(c.col)
    val rowGrads = Array.fill(rowFactors.numCols)(0.0)
    val colGrads = Array.fill(colFactors.numCols)(0.0)
    // compute the error for the cell
    val err = error(c)
    // do the rows first
    for (k <- 0 until rowFactors.numCols) {
      rowGrads(k) = -2.0 * weight() * err * col(k)
    }
    grads(rowFactors) = (c.row -> rowGrads)
    // then do the cols
    for (k <- 0 until colFactors.numCols) {
      colGrads(k) = -2.0 * weight() * err * row(k)
    }
    grads(colFactors) = (c.col -> colGrads)
    grads
  }
}

/**
 * Introduces biases for rows and columns, but otherwise identical to L2DotTerm
 */
trait L2WithBias extends DotTermWithBias with L2 {
  override def gradient(c: Cell): Gradients = {
    val grads = super.gradient(c)
    val g = -2.0 * weight() * error(c)
    grads(rowBias) = (c.row -> Array(g))
    grads(colBias) = (c.col -> Array(g))
    grads
  }
}

/**
 * Logistic loss suitable for binary matrices. The term is optimized by minimizing negative log-likelihood.
 */
trait Logistic extends DotTerm {

  // log prob of c.value
  def error(c: Cell) = {
    assert(c.value.double >= 0.0 || c.value.double <= 1.0)
    val score = dot(c)
    val logZ = log(exp(score) + 1.0)
    val lprob = score - logZ
    val liprob = -logZ // log(1) - logZ
    -(c.value.double * lprob + (1.0 - c.value.double) * liprob) // negative log likelihood
  }

  // for stochastic estimation, the value for a cell
  def value(c: Cell): Double = if (c.inMatrix == target) {
    error(c)
  } else 0.0

  def gradient(c: Cell): Gradients = {
    val grads = new Gradients
    val row = rowFactors.r(c.row)
    val col = colFactors.r(c.col)
    val rowGrads = Array.fill(rowFactors.numCols)(0.0)
    val colGrads = Array.fill(colFactors.numCols)(0.0)
    val score = dot(c)
    val escore = exp(score)
    val prob = escore / (escore + 1.0)
    val direction = -(c.value.double - prob) // gradient of negative log-likelihood
    // do the rows first
    for (k <- 0 until rowFactors.numCols) {
      rowGrads(k) = weight() * direction * col(k)
    }
    grads(rowFactors) = (c.row -> rowGrads)
    // then do the cols
    for (k <- 0 until colFactors.numCols) {
      colGrads(k) = weight() * direction * row(k)
    }
    grads(colFactors) = (c.col -> colGrads)
    grads
  }
}

/**
 * Include bias in the dot term, but otherwise same as LogisticDotTerm
 */
trait LogisticWithBias extends DotTermWithBias with Logistic {
  override def gradient(c: Cell): Gradients = {
    val grads = super.gradient(c)
    val score = dot(c)
    val escore = exp(score)
    val prob = escore / (escore + 1.0)
    val direction = -(c.value.double - prob) // gradient of negative log-likelihood
    val g = weight() * direction
    grads(rowBias) = (c.row -> Array(g))
    grads(colBias) = (c.col -> Array(g))
    grads
  }
}

trait BinaryError extends Logistic {
  def threshold: Double = 0.5

  override def value(c: Cell): Double = {
    val score = dot(c)
    val escore = exp(score)
    val prob = escore / (escore + 1.0)
    if ((prob > threshold && c.value.double > 0.5) || (prob <= threshold && c.value.double <= 0.5))
      0.0
    else
      1.0
  }

  override def gradient(c: Cell): Gradients =
    throw new Error("Gradient not defined, use only for evaluation")
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