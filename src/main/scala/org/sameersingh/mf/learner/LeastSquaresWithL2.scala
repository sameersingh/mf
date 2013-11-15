package org.sameersingh.mf.learner

import org.sameersingh.mf.{DoubleDenseMatrix, Cell, Term, CollectiveFactorization}
import scala.collection.mutable

class Parameters {
  private val _map = new mutable.HashMap[String, Double]
  // Per factor parameter
  private def key(f:DoubleDenseMatrix, attr: String): String = "FACT:%s_%s" format(f.name, attr)
  def apply(f:DoubleDenseMatrix, attr: String, default: Double): Double = _map.getOrElse(key(f, attr), default)

  def l2RegCoeff(f:DoubleDenseMatrix) = apply(f, "L2RegCoeff", 1.0)
  // Per term parameter
  private def key(t:Term, attr: String): String = "TERM:%s_%s" format(t.target.name, attr)
  def apply(t:Term, attr: String, default: Double): Double = _map.getOrElse(key(t, attr), default)

  def termWeight(t:Term) = apply(t, "Weight", 1.0)
}

class LeastSquaresWithL2(val t: Term, params: Parameters) {
  val numFactors = t.rowFactors.numCols
  val weight = params.termWeight(t)
  val rowRegCoeff = params.l2RegCoeff(t.rowFactors)
  val colRegCoeff = params.l2RegCoeff(t.colFactors)
  val numCells = t.target.trainCells.size

  def l2error(c: Cell) = {
    val latent = t.rowFactors.r(c.row).zip(t.colFactors.r(c.col)).foldLeft(0.0)((s, uv) => s + uv._1*uv._2)
    StrictMath.pow(c.value.double - latent, 2.0)
  }

  def trainL2Error = {
    weight * t.target.trainCells.foldLeft(0.0)(_ + l2error(_))
  }

  def testL2Error = {
    weight * t.target.testCells.foldLeft(0.0)(_ + l2error(_))
  }

  /**
   * Perform one step of gradient descent
   * @param c observed cell for which to perform a gradient update
   */
  def sgd(c: Cell, stepSize: Double) {
    val (rg, cg) = stochasticGradient(c)
    val rowF = t.rowFactors.r(c.row)
    val colF = t.colFactors.r(c.col)
    for(k <- 0 until numFactors) {
      rowF(k) = rowF(k) - stepSize*rg(k)
      colF(k) = colF(k) - stepSize*cg(k)
    }
  }

  def stochasticGradient(c: Cell): (Array[Double], Array[Double]) = {
    val row = t.rowFactors.r(c.row)
    val col = t.colFactors.r(c.col)
    val rowGrads = Array.fill(numFactors)(0.0)
    val colGrads = Array.fill(numFactors)(0.0)
    // compute the error for the cell
    val err = c.value.double - row.zip(col).foldLeft(0.0)((s, uv) => s + uv._1*uv._2)
    // do the rows first
    for(k <- 0 until numFactors) {
      var grad = 0.0
      grad += -2.0 * weight * err * col(k)
      grad += 2.0 * rowRegCoeff * (1.0/numCells) * row(k)
      rowGrads(k) = grad
    }
    // then do the cols
    for(k <- 0 until numFactors) {
      var grad = 0.0
      grad += -2.0 * weight * err * row(k)
      grad += 2.0 * colRegCoeff * (1.0/numCells) * col(k)
      colGrads(k) = grad
    }
    (rowGrads, colGrads)
  }

  def batch(stepSize: Double) {
    // create gradient vectors
    val rowGrads = new DoubleDenseMatrix(t.rowFactors.name+"_gradient", t.rowFactors.numCols)
    for(r <- t.rowFactors.rids) rowGrads += (r)
    val colGrads = new DoubleDenseMatrix(t.colFactors.name+"_gradient", t.colFactors.numCols)
    for(r <- t.colFactors.rids) colGrads += (r)
    // compute and accumulate gradients
    for(c <- t.target.trainCells) {
      val (rg, cg) = stochasticGradient(c)
      val rowG = rowGrads.r(c.row)
      val colG = colGrads.r(c.col)
      for(k <- 0 until numFactors) {
        rowG(k) = rowG(k) + rg(k)
        colG(k) = colG(k) + cg(k)
      }
    }
    // update the parameters
    for(r <- t.rowFactors.rids) {
      val row = t.rowFactors.r(r)
      val rowG = rowGrads.r(r)
      for(k <- 0 until numFactors) {
        row(k) = row(k) - stepSize*rowG(k)
      }
    }
    for(r <- t.colFactors.rids) {
      val col = t.colFactors.r(r)
      val colG = colGrads.r(r)
      for(k <- 0 until numFactors) {
        col(k) = col(k) - stepSize*colG(k)
      }
    }

  }

}
