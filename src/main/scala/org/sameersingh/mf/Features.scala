package org.sameersingh.mf

import scala.collection.mutable
import cc.factorie.la.{GrowableSparseBinaryTensor1, Tensor}
import cc.factorie.variable.{CategoricalVectorDomain, CategoricalVectorVariable}

/**
 * @author sameer
 * @since 5/22/14.
 */
class Features(val domain: CategoricalVectorDomain[String] = new CategoricalVectorDomain[String] {}) extends collection.mutable.Map[ID, Tensor] {
  type FType = CategoricalVectorVariable[String]

  private val features = new mutable.HashMap[ID, Tensor]

  override def get(key: ID): Option[Tensor] = features.get(key)

  def update(id: ID, feats: FType) {
    assert(feats.domain == domain)
    update(id, feats.value)
  }

  def update(id: ID, feats: Seq[String]) {
    domain.dimensionDomain ++= feats
    val f = new GrowableSparseBinaryTensor1(domain.dimensionDomain)
    f ++= feats.map(f => domain.dimensionDomain.getIndex(f))
    this.update(id, f)
  }

  def update(id: ID, singleFeat: String) {
    val f = features.getOrElseUpdate(id, new GrowableSparseBinaryTensor1(domain.dimensionDomain))
    f +=(domain.dimensionDomain.getIndex(singleFeat), 1.0)
  }

  def getName(idx: Int): String = domain.dimensionDomain.apply(idx).category

  def getIndex(string: String): Int = domain.dimensionDomain.getIndex(string)

  override def iterator: Iterator[(ID, Tensor)] = features.iterator

  override def -=(key: ID): this.type = {
    features -= key;
    this
  }

  override def +=(kv: (ID, Tensor)): this.type = {
    features += kv;
    this
  }
}
