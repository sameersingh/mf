package org.sameersingh.mf

import scala.util.Random
import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import java.util.zip.GZIPInputStream
import java.io.FileInputStream

/**
 * Load a matrix from a tab-separated file
 * @author sameer
 */
class MatrixLoader {

  def loadFile(filename: String, matrixName: String, trainingProp: Double, gzip: Boolean = false)(implicit cache: Cache, random: Random): ObservedMatrix = {
    val stream = if(gzip) new GZIPInputStream(new FileInputStream(filename)) else new FileInputStream(filename)
    val source = Source.fromInputStream(stream, "UTF-8")
    val cellArray = new ArrayBuffer[Cell]
    val matrix = new ObservedMatrix {
      val cells: Seq[Cell] = cellArray

      lazy val trainCells: Seq[Cell] = super.trainCells

      lazy val testCells: Seq[Cell] = super.testCells

      lazy val rowIDs: Set[ID] = super.rowIDs

      lazy val colIDs: Set[ID] = super.colIDs

      val name: String = matrixName
    }
    for(line <- source.getLines()) {
      val split = line.split('\t')
      assert(split.length > 0)
      val rowId = CachedID(split(0), matrixName + "_rows")
      for(c <- split.drop(1)) {
        val colId = CachedID(c, matrixName + "_cols")
        cellArray += new Cell {
          val isTrain: Boolean = random.nextDouble < trainingProp

          val value: Val = DoubleValue(1.0)

          val col: ID = colId

          val row: ID = rowId

          val inMatrix: ObservedMatrix = matrix
        }
      }
    }
    source.close()
    matrix
  }

}

object MatrixLoader extends MatrixLoader