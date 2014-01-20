package org.sameersingh.mf.learner

import org.sameersingh.mf._
import scala.util.Random

trait Trainer {
  def round(initStepSize: Double): Unit

  def trainCells(t: ObservedMatrix) = t.trainCells
}

class SGDTrainer(val targets: Seq[ObservedMatrix], val terms: Seq[Term]) extends Trainer {

  val params = terms.flatMap(_.params).distinct

  def update(c: Cell, stepSize: Double) {
    val grads = new Gradients
    terms.map(t => t.gradient(c)).foreach(gs => grads += gs)
    for (p <- params) p.gradientUpdate(grads, stepSize)
  }

  def round(initStepSize: Double) {
    var stepSize = initStepSize
    for (t <- targets) {
      for (c <- trainCells(t)) {
        update(c, stepSize)
      }
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

  val random: Random = new Random(0)

  def numSampledCells(t: ObservedMatrix): Int = 100

  def defaultValue: Val = DoubleValue(0.0)

  def randomSampledCell(t: ObservedMatrix): Cell = {
    val r = t.rowIDs.toSeq(random.nextInt(t.rowIDs.size))
    val c = t.colIDs.toSeq(random.nextInt(t.colIDs.size))
    new Cell {
      val row: ID = r

      val col: ID = c

      val value: Val = defaultValue

      val isTrain: Boolean = true

      val inMatrix: ObservedMatrix = t
    }
  }

  def randomSampledCells(t: ObservedMatrix, count: Int): Seq[Cell] = (0 until numSampledCells(t)).map(i => randomSampledCell(t))

  override def trainCells(t: ObservedMatrix): Seq[Cell] = {
    super.trainCells(t) ++ randomSampledCells(t, numSampledCells(t))
  }
}