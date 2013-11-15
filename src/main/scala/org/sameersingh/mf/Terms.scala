package org.sameersingh.mf

trait Term {
  def params: Seq[Parameters]
  // for stochastic estimation, the value for a cell
  def value(c: Cell): Double
  def gradient(c: Cell): Gradients
}

class DotTerm(val rowFactors: DoubleDenseMatrix,
              val colFactors: DoubleDenseMatrix,
              val weight: ParamDouble,
              val target: ObservedMatrix)
  extends Term {

  def this(params: ParameterSet, u: String, v: String, w: Double, target: ObservedMatrix) =
    this(params(u), params(v), params(target, "weight", w), target)

  assert(rowFactors.numCols == colFactors.numCols, "Number of columns for DotTerms should match %s->%d, %s->%d" format(rowFactors.name, rowFactors.numCols, colFactors.name, colFactors.numCols))

  private val _params: Seq[Parameters] = Seq(rowFactors, colFactors)

  def params = _params

  def dot(c: Cell) = rowFactors.r(c.row).zip(colFactors.r(c.col)).foldLeft(0.0)((s, uv) => s + uv._1*uv._2)

  def error(c: Cell) = c.value.double - dot(c)

  // for stochastic estimation, the value for a cell
  def value(c: Cell): Double =  if (c.inMatrix == target) {
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
    for(k <- 0 until rowFactors.numCols) {
      rowGrads(k) = -2.0 * weight() * err * col(k)
    }
    grads(rowFactors) = (c.row -> rowGrads)
    // then do the cols
    for(k <- 0 until colFactors.numCols) {
      colGrads(k) = -2.0 * weight() * err * row(k)
    }
    grads(colFactors) = (c.col -> colGrads)
    grads
  }
}

class DotTermWithBias(rowFactors: DoubleDenseMatrix,
                      colFactors: DoubleDenseMatrix,
                      val rowBias: ParamVector,
                      val colBias: ParamVector,
                      weight: ParamDouble,
                      target: ObservedMatrix) extends DotTerm(rowFactors, colFactors, weight, target) {
  def this(params: ParameterSet, u: String, v: String, w: Double, target: ObservedMatrix) =
    this(params(u), params(v), params.f(u, "bias"), params.f(v, "bias"), params(target, "weight", w), target)

  private val _params: Seq[Parameters] = super.params ++ Seq(rowBias, colBias)

  override def params: Seq[Parameters] = _params

  override def error(c: Cell): Double = super.error(c) - rowBias(c.row) - colBias(c.col)

  override def gradient(c: Cell): Gradients = {
    val grads = super.gradient(c)
    val g = -2.0 * weight() * error(c)
    grads(rowBias) = (c.row -> Array(g))
    grads(colBias) = (c.col -> Array(g))
    grads
  }
}

class L2Regularization(val factors: DoubleDenseMatrix, val weight: ParamDouble, val numCells: Int = 1)
  extends Term {
  def this(params: ParameterSet, f: String, n: Int) = this(params(f), params.l2RegCoeff(f), n)

  val params: Seq[Parameters] = Seq(factors)

  // for stochastic estimation, the value for a cell
  def value(c: Cell): Double = if(c.row.domain == factors.name) {
    (1.0/numCells)*weight()*factors.r(c.row).foldLeft(0.0)((s,u) => s + u*u)
  } else if(c.col.domain == factors.name) {
    (1.0/numCells)*weight()*factors.r(c.col).foldLeft(0.0)((s,v) => s + v*v)
  } else 0.0

  def gradient(c: Cell): Gradients = {
    val grads = new Gradients
    if(c.row.domain == factors.name) {
      val gs = Array.fill(factors.numCols)(0.0)
      val row = factors.r(c.row)
      for(k <- 0 until factors.numCols) {
        gs(k) = 2.0 * weight() * (1.0/numCells) * row(k)
      }
      grads(factors) = (c.row -> gs)
    } else if(c.col.domain == factors.name) {
      val gs = Array.fill(factors.numCols)(0.0)
      val col = factors.r(c.col)
      for(k <- 0 until factors.numCols) {
        gs(k) = 2.0 * weight() * (1.0/numCells) * col(k)
      }
      grads(factors) = (c.col -> gs)
    }
    grads
  }
}