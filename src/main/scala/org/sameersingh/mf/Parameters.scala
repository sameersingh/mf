package org.sameersingh.mf

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import java.io._
import java.util.zip.GZIPOutputStream
import scala.Some
import scala.io.Source

trait Parameters {
  type StochasticGradientType

  def name: String

  def gradientUpdate(grads: Gradients, stepSize: Double) {
    val gs = grads(this)
    for (g <- gs) {
      val gds = g.asInstanceOf[StochasticGradientType]
      gradientUpdate(gds, stepSize)
    }
  }

  def gradientUpdate(sgd: StochasticGradientType, stepSize: Double): Unit
}

class ParamDouble(val name: String, default: Double) extends Parameters {
  type StochasticGradientType = Double

  var value = default

  def apply(): Double = value

  def update(d: Double) = value = d

  def gradientUpdate(sgd: Double, stepSize: Double): Unit = value += stepSize * sgd
}

abstract class DenseMatrix[D](val name: String, val numCols: Int, init: () => D)(implicit m: ClassTag[D]) extends Parameters {

  val rowIDToRowPosition = new mutable.LinkedHashMap[ID, Int]()
  val rows = new ArrayBuffer[Array[D]]

  def +=(rid: ID): Array[D] = {
    rowIDToRowPosition(rid) = rows.length
    rows += Array.fill(numCols)(init())
    rows.last
  }

  def get(rid: ID): Option[Array[D]] = {
    rowIDToRowPosition.get(rid).map(i => rows(i))
  }

  def get(ridx: Int): Option[Array[D]] = {
    if (ridx < rows.length) Some(rows(ridx)) else None
  }

  def r(rid: ID): Array[D] = {
    val opt = get(rid)
    if (opt.isDefined)
      opt.get
    else this += rid
  }

  def r(ridx: Int): Array[D] = {
    val opt = get(ridx)
    if (opt.isDefined)
      opt.get
    else Array.fill(numCols)(init())
  }

  def ridxs = (0 until rows.length)

  def rids = rowIDToRowPosition.keys

  def c(k: Int): Seq[D] = {
    rows.map(r => r(k)).toSeq
  }

  def apply(rid: ID, k: Int): D = {
    assert(k < numCols, "Requesting k=%d where number of columns is %d" format(k, numCols))
    val opt = get(rid)
    assert(opt.isDefined, "Could not find value for (%s, %d)" format(rid, k))
    opt.get(k)
  }

  def update(rid: ID, k: Int, d: D): Unit = {
    assert(k < numCols, "Requesting k=%d where number of columns is %d" format(k, numCols))
    val opt = get(rid)
    if (opt.isDefined) {
      opt.get(k) = d
    } else {
      val arr = this += rid
      arr(k) = d
    }
  }

  override def toString = {
    val sb = new StringBuffer()
    sb.append("%s\t%s\n" format(name, (0 until numCols).mkString("\t")))
    for ((rid, ridx) <- rowIDToRowPosition) {
      sb.append("%s\t%s\n" format(rid, r(ridx).mkString("\t")))
    }
    sb.toString
  }
}

class DoubleDenseMatrix(name: String, numCols: Int, init: () => Double = () => 0.0) extends DenseMatrix[Double](name, numCols, init) {
  type StochasticGradientType = (ID, Array[Double])

  def gradientUpdate(sgd: (ID, Array[Double]), stepSize: Double) {
    val arr = r(sgd._1)
    for (k <- 0 until numCols)
      arr(k) = arr(k) - stepSize * sgd._2(k)
  }

}

class ParamVector(name: String, init: () => Double = () => 0.0) extends DoubleDenseMatrix(name, 1, init) {
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
    assert(_map.contains(name), "Parameter %s missing in parameter set" format (name))
    _map(name).asInstanceOf[T]
  }

  def update(f: DoubleDenseMatrix, attr: String, d: Double) = _map(factorKey(f.name, attr)) = new ParamDouble(factorKey(f.name, attr), d)

  def update(f: DoubleDenseMatrix, attr: String, v: DoubleDenseMatrix): Unit = _map(factorKey(f.name, attr)) = v

  def update(f: DoubleDenseMatrix, attr: String, init: () => Double): Unit = update(f, attr, new ParamVector(factorKey(f.name, attr), init))

  private def factorKey(fname: String, attr: String): String = "FACT:%s_%s" format(fname, attr)

  private def obsKey(tname: String, attr: String): String = "OBS:%s_%s" format(tname, attr)

  def f[T <: Parameters](fname: String, attr: String): T = _map(factorKey(fname, attr)).asInstanceOf[T]

  def apply[T <: Parameters](factor: DoubleDenseMatrix, attr: String): T = f(factor.name, attr)

  def apply[T <: Parameters](f: DoubleDenseMatrix, attr: String, default: => T): T = _map.getOrElseUpdate(factorKey(f.name, attr), default).asInstanceOf[T]

  def apply(f: DoubleDenseMatrix, attr: String, default: Double): ParamDouble =
    _map.getOrElseUpdate(factorKey(f.name, attr), new ParamDouble(factorKey(f.name, attr), default)).asInstanceOf[ParamDouble]

  // example
  def l2RegCoeff(f: DoubleDenseMatrix) = apply(f, "L2RegCoeff", 1.0)

  def l2RegCoeff(fname: String) = apply(apply[DoubleDenseMatrix](fname), "L2RegCoeff", 1.0)

  def t[T <: Parameters](tname: String, attr: String) = _map(obsKey(tname, attr)).asInstanceOf[T]

  def apply[T <: Parameters](target: ObservedMatrix, attr: String) = t(target.name, attr)

  def apply(t: ObservedMatrix, attr: String, default: Double): ParamDouble =
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

  def apply[P <: Parameters](p: P): Seq[P#StochasticGradientType] =
    _map.getOrElse(p.name, Seq.empty).map(_.asInstanceOf[P#StochasticGradientType]).toSeq

  def update[P <: Parameters, G <: P#StochasticGradientType](p: P, g: G) =
    _map.getOrElseUpdate(p.name, new ArrayBuffer) += g

  def clear[P <: Parameters](p: P) {
    _map.remove(p.name)
  }

  def +=(gs: Gradients): Gradients = {
    gs._map.foreach(p_gs => _map.getOrElseUpdate(p_gs._1, new ArrayBuffer) ++= p_gs._2)
    this
  }
}

object DoubleDenseMatrix {
  def save(m: DoubleDenseMatrix, filename: String, gzip: Boolean): Unit = save(m, if (gzip) new GZIPOutputStream(new FileOutputStream(filename)) else new FileOutputStream(filename))

  def save(m: DoubleDenseMatrix, os: OutputStream): Unit = {
    val writer = new PrintWriter(new BufferedOutputStream(os))
    writer.println(m.name + "\t" + m.numCols)
    for (r <- m.rids) {
      writer.print(r.id)
      for (c <- m.r(r)) {
        writer.print("\t" + c)
      }
      writer.println()
    }
    writer.flush()
    writer.close()
  }

  def load(is: InputStream, strToId: (String) => ID): DoubleDenseMatrix = {
    val source = Source.fromInputStream(is)
    var ddm: DoubleDenseMatrix = null
    for (line <- source.getLines()) {
      val split = line.split("\t")
      if (ddm == null) {
        assert(split.length == 2)
        ddm = new DoubleDenseMatrix(split(0), split(1).toInt)
      } else {
        assert(split.length == ddm.numCols + 1)
        val r = strToId(split(0))
        for(k <- 0 until ddm.numCols) {
          ddm(r, k) = split(k + 1).toDouble
        }
      }
    }
    source.close()
    ddm
  }
}