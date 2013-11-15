package org.sameersingh.mf

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class DenseMatrix[D](val name: String, val numCols: Int, init: => D)(implicit m: ClassTag[D]) {
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

class DoubleDenseMatrix(name: String, numCols: Int) extends DenseMatrix[Double](name, numCols, 0.0)
