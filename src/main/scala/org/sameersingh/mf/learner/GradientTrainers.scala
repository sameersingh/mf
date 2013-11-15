package org.sameersingh.mf.learner

import org.sameersingh.mf.{Gradients, ObservedMatrix, Term}


class SGDTrainer(val targets: Seq[ObservedMatrix], val terms: Seq[Term]) {

  val params = terms.flatMap(_.params).distinct

  def round(initStepSize: Double) {
    var stepSize = initStepSize
    for(t <- targets) {
      for (c <- t.trainCells) {
        val grads = terms.map(t => t.gradient(c)).foldLeft(new Gradients)((all, gs) => all += gs)
        for(p <- params) p.stochasticUpdate(grads, stepSize)
      }
    }
  }

}

class BatchTrainer(val targets: Seq[ObservedMatrix], val terms: Seq[Term]) {

  val params = terms.flatMap(_.params).distinct

  def round(initStepSize: Double) {
    var stepSize = initStepSize
    val grads = new Gradients
    for(t <- targets) {
      for (c <- t.trainCells) {
        terms.map(t => t.gradient(c)).foldLeft(grads)((all, gs) => all += gs)
      }
    }
    for(p <- params) p.stochasticUpdate(grads, stepSize)
  }

}