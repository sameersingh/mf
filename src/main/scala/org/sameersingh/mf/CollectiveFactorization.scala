package org.sameersingh.mf

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable

class CollectiveFactorization {
  val params = new ParameterSet
  val targets = new mutable.LinkedHashMap[String, ObservedMatrix]
  val terms = new ArrayBuffer[Term]

  def +=(target: ObservedMatrix) {
    assert(!targets.contains(target.name), "Matrix already exists with name: " + target.name)
    targets(target.name) = target
  }

  def +=(t: Term) {
    terms += t
  }

}
