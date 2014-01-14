package org.sameersingh.mf

import scala.util.Random
import org.sameersingh.mf.learner.SGDTrainer

/**
 * @author sameer
 */
object FactorizeMatrix extends App {
  implicit val random = new Random(0)
  implicit val cache = new Cache
  val filename = "/Users/sameer/Work/data/mf/tac12.th3.profiles.gz"
  val m = MatrixLoader.loadFile(filename, "M", 0.8, true)
  println(m)
  val numComps = 100
  val params = new ParameterSet
  params += new DoubleDenseMatrix("r", numComps, () => random.nextGaussian() / 100.0)
  params += new DoubleDenseMatrix("c", numComps, () => random.nextGaussian() / 100.0)
  params(params[DoubleDenseMatrix]("r"), "L2RegCoeff") = 0.1
  params(params[DoubleDenseMatrix]("c"), "L2RegCoeff") = 0.1
  val term = new DotTerm(params, "r", "c", 1.0, m)
  val l2r = new L2Regularization(params, "r", m.trainCells.size)
  val l2c = new L2Regularization(params, "c", m.trainCells.size)
  val terms = Seq(term, l2r, l2c)

  val trainer = new SGDTrainer(Seq(m), terms)
  for (i <- 0 until 500) {
    println("-----------------------")
    println("        ROUND %d" format (i + 1))
    println("-----------------------")
    println("Training value : " + term.avgTrainingValue(m))
    println("Test value     : " + term.avgTestValue(m))
    println("L2 R value     : " + l2r.value / l2r.weight())
    println("L2 C value     : " + l2c.value / l2c.weight())
    //*/
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
