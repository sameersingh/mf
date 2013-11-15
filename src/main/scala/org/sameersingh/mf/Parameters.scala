package org.sameersingh.mf

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

trait Parameters {
  type StochasticGradientType
  def name: String
  def stochasticUpdate(grads: Gradients, stepSize: Double) {
    val gs = grads(this)
    for(g <- gs) stochasticUpdate(g.asInstanceOf[StochasticGradientType], stepSize)
  }
  def stochasticUpdate(sgd: StochasticGradientType, stepSize: Double): Unit
}

class ParamDouble(val name: String, default: Double) extends Parameters {
  type StochasticGradientType = Double

  var value = default
  def apply() : Double = value
  def update(d: Double) = value = d

  def stochasticUpdate(sgd: Double, stepSize: Double): Unit = value += stepSize*sgd
}

abstract class DenseMatrix[D](val name: String, val numCols: Int, init: => D)(implicit m: ClassTag[D]) extends Parameters {

  val rowIDToRowPosition = new mutable.LinkedHashMap[ID, Int]()
  val rows = new ArrayBuffer[Array[D]]

  def +=(rid: ID) = {
    rowIDToRowPosition(rid) = rows.length
    rows += Array.fill(numCols)(init)
  }

  def get(rid: ID): Option[Array[D]] = {
    rowIDToRowPosition.get(rid).map(i => rows(i))
  }

  def get(ridx: Int): Option[Array[D]] = {
    if(ridx < rows.length) Some(rows(ridx)) else None
  }

  def r(rid: ID): Array[D] = {
    val opt = get(rid)
    assert(opt.isDefined, "Could not find value for row %s" format(rid))
    opt.get
  }

  def r(ridx: Int): Array[D] = {
    val opt = get(ridx)
    assert(opt.isDefined, "Could not find value for row idx %d" format(ridx))
    opt.get
  }

  def ridxs = (0 until rows.length)
  def rids = rowIDToRowPosition.keys

  def c(k : Int): Seq[D] = {
    rows.map(r => r(k)).toSeq
  }

  def apply(rid: ID, k: Int): D = {
    assert(k < numCols, "Requesting k=%d where number of columns is %d" format(k, numCols))
    val opt = get(rid)
    assert(opt.isDefined, "Could not find value for (%s, %d)" format(rid, k))
    opt.get(k)
  }
}

class DoubleDenseMatrix(name: String, numCols: Int) extends DenseMatrix[Double](name, numCols, 0.0) {
  type StochasticGradientType = (ID, Array[Double])

  def stochasticUpdate(sgd: (ID, Array[Double]), stepSize: Double) {
    val arr = r(sgd._1)
    for(k <- 0 until numCols)
      arr(k) = arr(k) + stepSize*sgd._2(k)
  }
}

class ParamVector(name: String) extends DoubleDenseMatrix(name, 1) {
  def apply(rid: ID): Double = apply(rid, 0)
  def update(rid: ID, d: Double) = r(rid)(0) = d
}

class ParameterSet extends Map[String, Parameters] {
  private val _map = new mutable.LinkedHashMap[String, Parameters]

  def get(key: String): Option[Parameters] = _map.get(key)

  def iterator: Iterator[(String, Parameters)] = _map.iterator

  def -(key: String): Map[String, Parameters] = _map.-(key).toMap

  def +[B1 >: Parameters](kv: (String, B1)): Map[String, B1] = _map.+(kv).toMap

  def apply[T <: Parameters](name: String): T = {
    assert(_map.contains(name), "Parameter %s missing in parameter set" format(name))
    _map(name).asInstanceOf[T]
  }

  private def factorKey(fname: String, attr: String): String = "FACT:%s_%s" format(fname, attr)
  private def obsKey(tname: String, attr: String): String = "OBS:%s_%s" format(tname, attr)

  def f[T <: Parameters](fname: String, attr: String): T = _map(factorKey(fname, attr)).asInstanceOf[T]
  def apply[T <: Parameters](factor:DoubleDenseMatrix, attr: String): T = f(factor.name, attr)
  def apply[T <: Parameters](f:DoubleDenseMatrix, attr: String, default: => T): T = _map.getOrElseUpdate(factorKey(f.name, attr), default).asInstanceOf[T]

  def apply(f:DoubleDenseMatrix, attr: String, default: Double): ParamDouble =
    _map.getOrElseUpdate(factorKey(f.name, attr), new ParamDouble(factorKey(f.name, attr), default)).asInstanceOf[ParamDouble]

  // example
  def l2RegCoeff(f:DoubleDenseMatrix) = apply(f, "L2RegCoeff", 1.0)
  def l2RegCoeff(fname: String) = apply(apply[DoubleDenseMatrix](fname), "L2RegCoeff", 1.0)

  def t[T <: Parameters](tname:String, attr: String) = _map(obsKey(tname, attr)).asInstanceOf[T]
  def apply[T <: Parameters](target:ObservedMatrix, attr: String) = t(target.name, attr)

  def apply(t:ObservedMatrix, attr: String, default: Double): ParamDouble =
    _map.getOrElseUpdate(obsKey(t.name, attr), new ParamDouble(obsKey(t.name, attr), default)).asInstanceOf[ParamDouble]

  def +=(factor: DoubleDenseMatrix) {
    assert(!contains(factor.name), "Factor already exists with name: " + factor.name)
    _map(factor.name) = factor
  }

  def +=(p: ParamVector) {
    assert(!contains(p.name), "Vector already exists with name: " + p.name)
    _map(p.name) = p
  }

}

class Gradients {
  private val _map = new mutable.LinkedHashMap[String, ArrayBuffer[Any]]

  def apply[P <: Parameters, G <: P#StochasticGradientType](p: P): Seq[G] =
    _map.getOrElse(p.name, Seq.empty).map(_.asInstanceOf[G]).toSeq
  def update[P <: Parameters, G <: P#StochasticGradientType](p: P, g: G) =
    _map.getOrElseUpdate(p.name, new ArrayBuffer) += g
  def clear[P <: Parameters](p: P) { _map.remove(p.name) }

  def +=(gs: Gradients): Gradients = {
    gs._map.foreach(p_gs => _map.getOrElseUpdate(p_gs._1, new ArrayBuffer) ++= p_gs._2)
    this
  }
}