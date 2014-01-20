package org.sameersingh.mf

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

  def rowIDs = cells.map(_.row).toSet

  def colIDs = cells.map(_.row).toSet

  override def toString = "%s\nTrain(%d):\t%s\nTest(%d):\t%s" format(name, trainCells.size, trainCells.take(5).mkString("\t"), testCells.size, testCells.take(5).mkString("\t"))
}
