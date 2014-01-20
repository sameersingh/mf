package org.sameersingh.mf

import org.junit._
import Assert._
import scala.collection.mutable.ArrayBuffer
import org.sameersingh.mf.learner.{Trainer, BatchTrainer, SGDTrainer}
import scala.util.Random

@Test
class LearningTest {

  implicit val random = new Random(0)

  def smallSingleMatrix(numComps: Int, noiseVar: Double = 0.0) = {
    val numRows = 10
    val numCols = 10
    val trainingProp = 0.5
    val rowFactors = new DoubleDenseMatrix("r_t", numComps)
    val colFactors = new DoubleDenseMatrix("c_t", numComps)
    for (k <- 0 until numComps) {
      for (i <- 0 until numRows) {
        rowFactors(SimpleID(i, "r"), k) = random.nextGaussian() / math.sqrt(numComps.toDouble)
      }
      for (j <- 0 until numCols) {
        colFactors(SimpleID(j, "c"), k) = random.nextGaussian() / math.sqrt(numComps.toDouble)
      }
    }
    //println("L2 R value     : " + new L2Regularization(rowFactors, new ParamDouble("w", 1.0)).value)
    //println("L2 C value     : " + new L2Regularization(colFactors, new ParamDouble("w", 1.0)).value)

    //println(rowFactors)
    //println(colFactors)

    val genCells = new ArrayBuffer[Cell]
    val matrix = new ObservedMatrix {
      val cells = genCells

      def name = "testMatrix"
    }
    for (i <- 0 until numRows)
      for (j <- 0 until numCols) {
        val rid = SimpleID(i, "r")
        val cid = SimpleID(j, "c")
        val doubleValue = rowFactors.r(rid).zip(colFactors.r(cid)).map(uv => uv._1 * uv._2).sum + noiseVar * random.nextGaussian()
        val cell = new Cell {
          val row = rid

          val col = cid

          val value = DoubleValue(doubleValue)

          val isTrain = random.nextDouble() < trainingProp

          val inMatrix = matrix
        }
        genCells += cell
      }
    matrix
  }

  def genParams(numComps: Int) = {
    val params = new ParameterSet
    params += new DoubleDenseMatrix("r", numComps, () => random.nextGaussian() / 100.0)
    params += new DoubleDenseMatrix("c", numComps, () => random.nextGaussian() / 100.0)
    params(params[DoubleDenseMatrix]("r"), "L2RegCoeff") = 0.1
    params(params[DoubleDenseMatrix]("c"), "L2RegCoeff") = 0.1
    params
  }

  def terms(m: ObservedMatrix) = {
    val params = genParams(1)
    val term = new DotTerm(params, "r", "c", 1.0, m) with L2
    val l2r = new L2Regularization(params, "r", m.trainCells.size)
    val l2c = new L2Regularization(params, "c", m.trainCells.size)
    (term, l2r, l2c)
  }

  def train(trainer: Trainer, term: DotTerm, l2r: L2Regularization, l2c: L2Regularization, m: ObservedMatrix) {
    //println("Init Training value : " + term.avgTrainingValue(m))
    //println("Init Test value     : " + term.avgTestValue(m))
    for (i <- 0 until 500) {
      /*println("-----------------------")
      println("        ROUND %d" format (i + 1))
      println("-----------------------")
      println("Training value : " + term.avgTrainingValue(m))
      println("Test value     : " + term.avgTestValue(m))
      println("L2 R value     : " + l2r.value / l2r.weight())
      println("L2 C value     : " + l2c.value / l2c.weight())
      */
      /*
      println("Row Factors:")
      println(params("r"))
      println("Col Factors:")
      println(params("c"))
      */
      trainer.round(0.01)
    }
    println("Final Training value : " + term.avgTrainingValue(m))
    println("Final Test value     : " + term.avgTestValue(m))
  }

  @Test
  def batchTrainingSmallSingle() {
    for (i <- 0 until 5) {
      println(" --- Batch %d ---" format(i+1))
      val m = smallSingleMatrix(1, 0.01)
      //println(m)
      val (term, l2r, l2c) = terms(m)
      val trainer = new BatchTrainer(Seq(m), Seq(term, l2r, l2c))
      train(trainer, term, l2r, l2c, m)
      assertEquals(0.0, term.avgTrainingValue(m), 0.001)
      assertEquals(0.0, term.avgTestValue(m), 0.0025)
    }
  }

  @Test
  def sgdTrainingSmallSingle() {
    for (i <- 0 until 5) {
      println(" --- SGD %d ---" format(i+1))
      val m = smallSingleMatrix(1, 0.01)
      //println(m)
      val (term, l2r, l2c) = terms(m)
      val trainer = new SGDTrainer(Seq(m), Seq(term, l2r, l2c))
      train(trainer, term, l2r, l2c, m)
      assertEquals(0.0, term.avgTrainingValue(m), 0.001)
      assertEquals(0.0, term.avgTestValue(m), 0.0025)
    }
  }
}
