package org.sameersingh.mf

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable


class Term(val rowFactors: DoubleDenseMatrix, val colFactors: DoubleDenseMatrix, val target: ObservedMatrix)

class CollectiveFactorization {
  val factors = new mutable.LinkedHashMap[String, DoubleDenseMatrix]
  val targets = new mutable.LinkedHashMap[String, ObservedMatrix]
  val terms = new ArrayBuffer[Term]

  def +=(factor: DoubleDenseMatrix) {
    assert(!factors.contains(factor.name), "Factor already exists with name: " + factor.name)
    factors(factor.name) = factor
  }

  def +=(target: ObservedMatrix) {
    assert(!targets.contains(target.name), "Matrix already exists with name: " + target.name)
    targets(target.name) = target
  }

  def +=(t: String, u: String, v: String) {
    val uf = factors(u)
    val vf = factors(v)
    assert(uf.numCols == vf.numCols,
      "The number of columns for factors %s (%d) and %s (%d) are not the same." format(u, uf.numCols, v, vf.numCols))
    terms += new Term(factors(u), factors(v), targets(t))
  }

}
