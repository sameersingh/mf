package org.sameersingh.mf

import scala.util.Random
import org.sameersingh.mf.learner.{Sampling, SGDTrainer}
import org.sameersingh.utils.timing.TimeUtil

/**
 * @author sameer
 */
object FactorizeMatrix extends App {
  implicit val random = new Random(0)
  implicit val cache = new Cache
  val baseDir = "/Users/sameer/Work/data/mf/"
  val filename = baseDir + "tac12.th3.profiles.gz"
  val rOutputFilename = baseDir + "mf_output/entity_profiles.gz"
  val cOutputFilename = baseDir + "mf_output/verb_profiles.gz"
  val pruneRows = 1
  val pruneCols = 1
  TimeUtil.init
  val original = MatrixLoader.loadFile(filename, "M", 0.8, true)
  TimeUtil.snapshot(original.toString)
  val m = Matrix.prune(original, pruneRows, pruneCols)
  val numZeroCells = (m.testCells.size / 10)
  val zeroCells = (0 until numZeroCells).map(i => Matrix.randomSampledCell(m, DoubleValue(0.0)))
  TimeUtil.snapshot(m.toString)

  val numComps = 100
  val r = new DoubleDenseMatrix("r", numComps, () => random.nextGaussian() / 100.0)
  val c = new DoubleDenseMatrix("c", numComps, () => random.nextGaussian() / 100.0)

  val params = new ParameterSet
  params += r
  params += c
  params(r, "bias") = (() => random.nextGaussian() / 100.0)
  params(c, "bias") = (() => random.nextGaussian() / 100.0)
  params(r, "L2RegCoeff") = 0.1
  params(c, "L2RegCoeff") = 0.1
  // objective terms
  val term = new DotTerm(params, "r", "c", 1.0, m) with Logistic
  val l2r = new L2Regularization(params, "r", m.trainCells.size)
  val l2c = new L2Regularization(params, "c", m.trainCells.size)
  val terms = Seq(term, l2r, l2c)
  val trainer = new SGDTrainer(Seq(m), terms) with Sampling

  // evaluation terms
  val nll = new DotTerm(params, "r", "c", 1.0, m) with Logistic
  val l2Error = new DotTerm(params, "r", "c", 1.0, m) with L2
  val binError = new DotTerm(params, "r", "c", 1.0, m) with BinaryError

  for (i <- 0 until 500) {
    println("-----------------------")
    println("        ROUND %d" format (i + 1))
    println("-----------------------")
    TimeUtil.snapshot("Training LL    : " + nll.avgTrainingValue(m))
    TimeUtil.snapshot("Test LL        : " + nll.avgTestValue(m))
    TimeUtil.snapshot("Test LL (0)    : " + nll.avgValue(zeroCells))
    //TimeUtil.snapshot("Training L2    : " + l2Error.avgTrainingValue(m))
    //TimeUtil.snapshot("Test L2        : " + l2Error.avgTestValue(m))
    //TimeUtil.snapshot("Test L2 (0)    : " + l2Error.avgValue(zeroCells))
    TimeUtil.snapshot("Training Err   : " + binError.avgTrainingValue(m))
    TimeUtil.snapshot("Test Err       : " + binError.avgTestValue(m))
    TimeUtil.snapshot("Test Err (0)   : " + binError.avgValue(zeroCells))
    TimeUtil.snapshot("L2 R value     : " + l2r.value / l2r.weight())
    TimeUtil.snapshot("L2 C value     : " + l2c.value / l2c.weight())
    //*/
    /*
    println("Row Factors:")
    println(params("r"))
    println("Col Factors:")
    println(params("c"))
    */
    trainer.round(0.01)
    TimeUtil.snapshot("Round done.")
    DoubleDenseMatrix.save(r, rOutputFilename + (i%5), true)
    DoubleDenseMatrix.save(c, cOutputFilename + (i%5), true)
    TimeUtil.snapshot("Matrices written")
  }
  println("-----------------------")
  println("        FINAL")
  println("-----------------------")
  TimeUtil.snapshot("Training LL    : " + nll.avgTrainingValue(m))
  TimeUtil.snapshot("Test LL        : " + nll.avgTestValue(m))
  TimeUtil.snapshot("Test LL (0)    : " + nll.avgValue(zeroCells))
  //TimeUtil.snapshot("Training L2    : " + l2Error.avgTrainingValue(m))
  //TimeUtil.snapshot("Test L2        : " + l2Error.avgTestValue(m))
  //TimeUtil.snapshot("Test L2 (0)    : " + l2Error.avgValue(zeroCells))
  TimeUtil.snapshot("Training Err   : " + binError.avgTrainingValue(m))
  TimeUtil.snapshot("Test Err       : " + binError.avgTestValue(m))
  TimeUtil.snapshot("Test Err (0)   : " + binError.avgValue(zeroCells))
  TimeUtil.snapshot("L2 R value     : " + l2r.value / l2r.weight())
  TimeUtil.snapshot("L2 C value     : " + l2c.value / l2c.weight())
}
