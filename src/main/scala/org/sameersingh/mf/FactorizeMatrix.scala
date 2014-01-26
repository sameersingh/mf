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
  val baseDir = "/Users/sameer/Work/data/entity-embedding/"
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
  val dotValue = new DotValue(params, "r", "c", m) with LogisticDot
  val term = new DotL2(params, dotValue, 1.0)
  val l2r = new L2Regularization(params, "r", m.trainCells.size)
  val l2c = new L2Regularization(params, "c", m.trainCells.size)
  val terms = Seq(term, l2r, l2c)
  val trainer = new SGDTrainer(Seq(m), terms) with Sampling

  // evaluation terms
  val nll = new NLL {
    def predictValue = dotValue
  }
  val l2 = new L2 {
    def predictValue = dotValue
  }
  val f1 = new PerCellF1 {
    def predictValue = dotValue
  }
  val hamming = new Hamming {
    def predictValue = dotValue
  }
  val eval = Evaluators(nll, f1, l2, hamming)

  for (i <- 0 until 500) {
    println("-----------------------")
    println("        ROUND %d" format (i + 1))
    println("-----------------------")
    TimeUtil.snapshot(eval.string(m, zeroCells))
    TimeUtil.snapshot("L2 R value     : " + l2r.value / l2r.weight())
    TimeUtil.snapshot("L2 C value     : " + l2c.value / l2c.weight())
    trainer.round(0.01)
    TimeUtil.snapshot("Round done.")
    DoubleDenseMatrix.save(r, rOutputFilename + (i % 5), true)
    DoubleDenseMatrix.save(c, cOutputFilename + (i % 5), true)
    TimeUtil.snapshot("Matrices written")
  }
  println("-----------------------")
  println("        FINAL")
  println("-----------------------")
  TimeUtil.snapshot(eval.string(m, zeroCells))
  TimeUtil.snapshot("L2 R value     : " + l2r.value / l2r.weight())
  TimeUtil.snapshot("L2 C value     : " + l2c.value / l2c.weight())
}
