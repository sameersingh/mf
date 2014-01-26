package org.sameersingh.mf.learner

import org.sameersingh.mf._
import scala.util.Random
import cc.factorie._

/**
 * Trainers that perform optimization on observed matrices. They are responsible for minimizing the objective function.
 *
 * @author sameer
 */
trait Trainer {
  def round(initStepSize: Double): Unit

  def trainCells(t: ObservedMatrix) = t.trainCells
}

class BatchedSGDTrainer(val batchSize: Int, val targets: Seq[ObservedMatrix], val terms: Seq[Term])(implicit val random: Random) extends Trainer {

  val params = terms.flatMap(_.params).distinct

  def updateBatch(cs: Seq[Cell], stepSize: Double) {
    val grads = new Gradients
    for (c <- cs) {
      terms.map(t => t.gradient(c)).foreach(gs => grads += gs)
    }
    for (p <- params) p.gradientUpdate(grads, stepSize)
  }

  def round(initStepSize: Double) {
    var stepSize = initStepSize
    for (batch <- targets.map(t => trainCells(t)).flatten.shuffle.grouped(batchSize)) {
      updateBatch(batch, stepSize)
    }
  }

}

class SGDTrainer(val targets: Seq[ObservedMatrix], val terms: Seq[Term])(implicit val random: Random) extends Trainer {

  val params = terms.flatMap(_.params).distinct

  def update(c: Cell, stepSize: Double) {
    val grads = new Gradients
    terms.map(t => t.gradient(c)).foreach(gs => grads += gs)
    for (p <- params) p.gradientUpdate(grads, stepSize)
  }

  def round(initStepSize: Double) {
    var stepSize = initStepSize
    for (c <- targets.map(t => trainCells(t)).flatten.shuffle) {
      update(c, stepSize)
    }
  }

}

class BatchTrainer(val targets: Seq[ObservedMatrix], val terms: Seq[Term]) extends Trainer {

  val params = terms.flatMap(_.params).distinct

  def round(initStepSize: Double) {
    var stepSize = initStepSize
    val grads = new Gradients
    for (t <- targets) {
      for (c <- trainCells(t)) {
        terms.map(t => t.gradient(c)).foreach(gs => grads += gs)
      }
    }
    for (p <- params) p.gradientUpdate(grads, stepSize)
  }
}

trait Sampling extends Trainer {

  def random: Random

  def numSampledCells(t: ObservedMatrix): Int = t.trainCells.size

  def defaultValue: Val = DoubleValue(0.0)

  def randomSampledCells(t: ObservedMatrix, count: Int): Seq[Cell] = (0 until count).map(i => Matrix.randomSampledCell(t, defaultValue)(random))

  override def trainCells(t: ObservedMatrix): Seq[Cell] = {
    super.trainCells(t) ++ randomSampledCells(t, numSampledCells(t))
  }
}