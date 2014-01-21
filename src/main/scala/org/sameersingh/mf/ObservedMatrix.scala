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

trait MutableObservedMatrix extends ArrayBuffer[Cell] with ObservedMatrix {
  def cells: Seq[Cell] = this
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
      val pruned = new MutableObservedMatrix {
        override lazy val trainCells: Seq[Cell] = super.trainCells

        override lazy val testCells: Seq[Cell] = super.testCells

        override lazy val rowIDs: Seq[ID] = super.rowIDs

        override lazy val colIDs: Seq[ID] = super.colIDs

        val name: String = m.name
      }

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
