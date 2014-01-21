package org.sameersingh.mf

import org.junit._
import Assert._
import scala.collection.mutable.ArrayBuffer

class MatrixTest {

  @Test
  def testPruning = {
    val numRows = 5
    val numCols = 5
    val m = new MutableObservedMatrix {
      def name: String = "m"
    }
    for (i <- 0 until numRows)
      for (j <- 0 until numCols) {
        if (i < j) {
          m += new Cell {
            val row: ID = SimpleID(i, "r")
            val col: ID = SimpleID(j, "c")
            val value: Val = DoubleValue(1.0)

            def isTrain: Boolean = true

            val inMatrix: ObservedMatrix = m
          }
        }
      }
    val pruned = Matrix.prune(m, 1, 1)
    assertEquals(m.cells.size, pruned.cells.size)
    val actuallyPruned = Matrix.prune(m, 2, 2)
    assertEquals(8, actuallyPruned.cells.size)
  }

}
