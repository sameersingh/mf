package org.sameersingh.mf

trait Cell {
  def row: ID
  def col: ID
  def value: Val
  def isTrain: Boolean
}

trait ObservedMatrix {
  def name: String
  def cells: Seq[Cell]
  def trainCells = cells.filter(_.isTrain)
  def testCells = cells.filterNot(_.isTrain)
}
