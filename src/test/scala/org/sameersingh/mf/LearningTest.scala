package org.sameersingh.mf

import org.junit._
import Assert._
import org.sameersingh.mf.learner._
import scala.util.Random

class LearningTest {

  implicit val random = new Random(0)

  def smallSingleMatrix(numComps: Int, noiseVar: Double = 0.0, f: Double => Double = x => x) = {
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

    val matrix = new Matrix("testMatrix")
    for (i <- 0 until numRows)
      for (j <- 0 until numCols) {
        val rid = SimpleID(i, "r")
        val cid = SimpleID(j, "c")
        val doubleValue = f(rowFactors.r(rid).zip(colFactors.r(cid)).map(uv => uv._1 * uv._2).sum + noiseVar * random.nextGaussian())
        val cell = new Cell {
          val row = rid

          val col = cid

          val value = DoubleValue(doubleValue)

          val isTrain = random.nextDouble() < trainingProp

          val inMatrix = matrix
        }
        matrix += cell
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
    val term = new DotL2(params, "r", "c", 1.0, m)
    val l2r = new L2Regularization(params, "r", m.trainCells.size)
    val l2c = new L2Regularization(params, "c", m.trainCells.size)
    (term, l2r, l2c)
  }

  def logTerms(m: ObservedMatrix) = {
    val params = genParams(1)
    val term = new LogisticDotNLL(new DotValue(params, "r", "c", m) with LogisticDot, params(m, "weight", 1.0))
    val l2r = new L2Regularization(params, "r", m.trainCells.size)
    val l2c = new L2Regularization(params, "c", m.trainCells.size)
    (term, l2r, l2c)
  }

  def train(trainer: Trainer, term: ErrorTerm, l2r: L2Regularization, l2c: L2Regularization, m: ObservedMatrix) {
    for (i <- 0 until 500) {
      trainer.round(0.01)
    }
    println(term.evalTrain(m).mkString("\n"))
    println(term.evalTest(m).mkString("\n"))
  }

  @Test
  def batchTrainingSmallSingle() {
    for (i <- 0 until 5) {
      println(" --- Batch %d ---" format (i + 1))
      val m = smallSingleMatrix(1, 0.01)
      //println(m)
      val (term, l2r, l2c) = terms(m)
      val trainer = new BatchTrainer(Seq(m), Seq(term, l2r, l2c))
      train(trainer, term, l2r, l2c, m)
      assertEquals(0.0, term.avgValue(m.trainCells), 0.001)
      //assertEquals(0.0, term.avgValue(m.testCells), 0.0025)
    }
  }

  @Test
  def sgdTrainingL2SmallSingle() {
    for (i <- 0 until 5) {
      println(" --- SGD L2 %d ---" format (i + 1))
      val m = smallSingleMatrix(1, 0.01)
      //println(m)
      val (term, l2r, l2c) = terms(m)
      val trainer = new SGDTrainer(Seq(m), Seq(term, l2r, l2c))
      train(trainer, term, l2r, l2c, m)
      assertEquals(0.0, term.avgValue(m.trainCells), 0.001)
      //assertEquals(0.0, term.avgValue(m.testCells), 0.035)
    }
  }

  @Test
  def sgdTrainingLogisticSmallSingle() {
    for (i <- 0 until 5) {
      println(" --- SGD Logistic %d ---" format (i + 1))
      val m = smallSingleMatrix(1, 0.0, x => Util.sigmoid(100*x))
      //println(m)
      val (term, l2r, l2c) = logTerms(m)
      val trainer = new SGDTrainer(Seq(m), Seq(term, l2r, l2c))
      train(trainer, term, l2r, l2c, m)
      assertEquals(0.0, term.avgValue(m.trainCells), 0.2)
      assertEquals(0.0, term.avgValue(m.testCells), 0.2)
    }
  }

  @Test
  def factorieLogisticTrainingSmallSingle() {
    for (i <- 0 until 1) {
      println(" --- Factorie Logistic %d ---" format (i + 1))
      // val m = smallSingleMatrix(3, 0.0, Util.sigmoid)
      val m = smallSingleMatrix(1, 0.0, x => if(random.nextDouble() < Util.sigmoid(x*100)) 1.0 else 0.0)
      //println(m)
      val fopt = new BasicFactorization(m, FactorizationConfig(1, 0.1, 1.0, 500, 100)) with LogisticLoss
      println(fopt.avgValue(m.trainCells))
      println(fopt.avgValue(m.testCells))
      fopt.optimize
      fopt.debug
      println(fopt.avgValue(m.trainCells))
      println(fopt.avgValue(m.testCells))
      // m.trainCells.foreach(c => println(fopt.pred(c) + "\t" + c.value + "\t" + fopt.loss(c)))
      assertEquals(0.0, fopt.avgValue(m.trainCells), 0.1)
      assertEquals(0.0, fopt.avgValue(m.testCells), 0.25)
    }
  }

  @Test
  def factorieL2TrainingSmallSingle() {
    for (i <- 0 until 5) {
      println(" --- Factorie L2 %d ---" format (i + 1))
      val m = smallSingleMatrix(1, 0.0)
      //println(m)
      val fopt = new BasicFactorization(m, FactorizationConfig(1, 0.1, 0.01, 500, 100)) with L2Loss
      println(fopt.avgValue(m.trainCells))
      fopt.optimize
      fopt.debug
      println(fopt.avgValue(m.trainCells))
      println(fopt.avgValue(m.testCells))
      //assertEquals(0.0, fopt.avgValue(m.testCells), 0.0025)
    }
  }

  @Test
  def factorieLogisticFourCells() {
    val m = new Matrix("testMatrix")
    m += new Cell {
      override val inMatrix: ObservedMatrix = m

      override def isTrain: Boolean = true

      override val value: Val = DoubleValue(0.0)
      override val col: ID = SimpleID(0, "c")
      override val row: ID = SimpleID(0, "r")
    }
    m += new Cell {
      override val inMatrix: ObservedMatrix = m

      override def isTrain: Boolean = true

      override val value: Val = DoubleValue(1.0)
      override val col: ID = SimpleID(1, "c")
      override val row: ID = SimpleID(0, "r")
    }
    m += new Cell {
      override val inMatrix: ObservedMatrix = m

      override def isTrain: Boolean = true

      override val value: Val = DoubleValue(0.0)
      override val col: ID = SimpleID(0, "c")
      override val row: ID = SimpleID(1, "r")
    }
    m += new Cell {
      override val inMatrix: ObservedMatrix = m

      override def isTrain: Boolean = true

      override val value: Val = DoubleValue(1.0)
      override val col: ID = SimpleID(1, "c")
      override val row: ID = SimpleID(1, "r")
    }
    val fopt = new BasicFactorization(m, FactorizationConfig(1, 0.0, 1.0, 500, 100)) with LogisticLoss
    fopt.optimize
    fopt.debug
    val loss = fopt.avgValue(m.trainCells)
    println(loss)
    assert(math.abs(loss) < 1.0e-4)
    val col0emb = fopt.params.colTensor(SimpleID(0, "c"))(0)
    val col1emb = fopt.params.colTensor(SimpleID(1, "c"))(0)
    val row0emb = fopt.params.rowTensor(SimpleID(0, "r"))(0)
    val row1emb = fopt.params.rowTensor(SimpleID(1, "r"))(0)
    println(col0emb + "\t" + col1emb)
    assertEquals(row0emb, row1emb, 1e-3)
    assert(if(row0emb > 0) col0emb < col1emb else col0emb > col1emb)
  }
}
