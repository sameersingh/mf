package org.sameersingh.mf.learner

import org.sameersingh.mf.{Cell, Gradients, ObservedMatrix, Term}

trait Trainer {
  def round(initStepSize: Double): Unit
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
      for (c <- t.trainCells) {
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
      for (c <- t.trainCells) {
        terms.map(t => t.gradient(c)).foreach(gs => grads += gs)
      }
    }
    for (p <- params) p.gradientUpdate(grads, stepSize)
  }

}