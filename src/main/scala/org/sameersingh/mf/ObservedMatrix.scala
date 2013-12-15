package org.sameersingh.mf

trait Cell {
  def row: ID

  def col: ID

  def value: Val

  def isTrain: Boolean

  def inMatrix: ObservedMatrix

  override def toString = "(%s, %s) = %f" format(row, col, value.double)
}

trait ObservedMatrix {
  def name: String

  def cells: Seq[Cell]

  def trainCells = cells.filter(_.isTrain)

  def testCells = cells.filterNot(_.isTrain)

  override def toString = "%s\nTrain(%d):\t%s\nTest(%d):\t%s" format(name, trainCells.size, trainCells.mkString("\t"), testCells.size, testCells.mkString("\t"))
}
