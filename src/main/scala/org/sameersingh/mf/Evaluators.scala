package org.sameersingh.mf

/**
 * Created by sameer on 1/25/14.
 */
class Evaluators(val evals: Seq[Evaluator]) {
  def string(m: ObservedMatrix, additionalTestCells: Seq[Cell]) = {
    val combinedTest = m.testCells ++ additionalTestCells
    (for (e <- evals) yield {
      val train = e.evalTrain(m)
      val test = e.evalTest(m)
      val additional = e.eval(additionalTestCells, "Additional ")
      val combined = e.eval(combinedTest, "Combined Test ")
      for (r <- train ++ test ++ additional ++ combined) yield (r._1 + "\t:\t" + r._2)
    }).flatten.mkString("\n")
  }
}

object Evaluators {
  def apply(evals: Evaluator*) = new Evaluators(evals)
}
