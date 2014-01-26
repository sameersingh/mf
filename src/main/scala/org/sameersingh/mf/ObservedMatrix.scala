package org.sameersingh.mf

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable
import scala.util.Random
import com.typesafe.scalalogging.slf4j.Logging

trait Cell {
  val row: ID

  val col: ID

  val value: Val

  def isTrain: Boolean

  val inMatrix: ObservedMatrix

  override def toString = "(%s, %s) = %f" format(row, col, value.double)
}

trait ObservedMatrix {
  def name: String

  def cells: Seq[Cell]

  def trainCells = cells.filter(_.isTrain)

  def testCells = cells.filterNot(_.isTrain)

  def rowIDs = cells.map(_.row).toSet.toSeq

  def colIDs = cells.map(_.col).toSet.toSeq

  override def toString = "%s\nTrain(%d):\t%s\nTest(%d):\t%s\nRows(%d):\t%s\nCols(%d):\t%s" format(name,
    trainCells.size, trainCells.take(5).mkString("\t"),
    testCells.size, testCells.take(5).mkString("\t"),
    rowIDs.size, rowIDs.take(5).mkString("\t"),
    colIDs.size, colIDs.take(5).mkString("\t"))
}

trait GrowableObservedMatrix extends ObservedMatrix {
  def +=(c: Cell): Unit

  def ++=(cs: Traversable[Cell]): Unit = cs.foreach(this += _)
}

class Matrix(val name: String) extends GrowableObservedMatrix {
  protected val _cells = new ArrayBuffer[Cell]
  protected val _trainCells = new ArrayBuffer[Cell]
  protected val _testCells = new ArrayBuffer[Cell]
  protected val _rowIDs = new mutable.LinkedHashSet[ID]
  protected val _colIDs = new mutable.LinkedHashSet[ID]

  def cells: Seq[Cell] = _cells.toSeq

  def +=(c: Cell): Unit = {
    _cells += c
    if (c.isTrain) _trainCells += c
    else _testCells += c
    _rowIDs += c.row
    _colIDs += c.col
  }

  override lazy val trainCells: Seq[Cell] = _trainCells

  override lazy val testCells: Seq[Cell] = _testCells

  override lazy val rowIDs: Seq[ID] = _rowIDs.toSeq

  override lazy val colIDs: Seq[ID] = _colIDs.toSeq
}

object Matrix extends Logging {
  def randomSampledCell(t: ObservedMatrix, v: Val)(implicit random: Random): Cell = {
    val r = t.rowIDs.toSeq(random.nextInt(t.rowIDs.size))
    val c = t.colIDs.toSeq(random.nextInt(t.colIDs.size))
    new Cell {
      val row: ID = r

      val col: ID = c

      val value: Val = v

      val isTrain: Boolean = true

      val inMatrix: ObservedMatrix = t
    }
  }

  def prune(m: ObservedMatrix, minCellsInRow: Int = 5, minCellsInCol: Int = 5): ObservedMatrix = {
    val rowCounts = new mutable.LinkedHashMap[ID, Int]()
    val colCounts = new mutable.LinkedHashMap[ID, Int]()
    for (c <- m.cells) {
      rowCounts(c.row) = rowCounts.getOrElse(c.row, 0) + 1
      colCounts(c.col) = colCounts.getOrElse(c.col, 0) + 1
    }
    if (rowCounts.minBy(_._2)._2 >= minCellsInRow && colCounts.minBy(_._2)._2 >= minCellsInCol) {
      logger.info("Not pruning.")
      m
    } else {
      val keepRows = rowCounts.filter(p => p._2 >= minCellsInRow).map(_._1).toSet
      val keepCols = colCounts.filter(p => p._2 >= minCellsInCol).map(_._1).toSet
      val pruned = new Matrix(m.name)

      pruned ++= m.cells.filter(c => keepCols(c.col) && keepRows(c.row)).map(c => new Cell {
        val row: ID = c.row
        val col: ID = c.col
        val value: Val = c.value

        def isTrain: Boolean = c.isTrain

        val inMatrix: ObservedMatrix = pruned
      })

      pruned
    }
  }
}
