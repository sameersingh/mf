package org.sameersingh.mf

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

trait ID {
  def id: String
  def idx: Int
  def apply = id
  def domain: String
}

trait Val {
  def double: Double
  def apply = double
}

case class SimpleID(idx: Int, domain: String) extends ID {

  def id: String = idx.toString()

  override def toString = id
}

case class DoubleValue(double: Double) extends Val

class Cache {
  val _map = new mutable.HashMap[String, mutable.HashMap[String, Int]]
  val _keys = new mutable.HashMap[String, ArrayBuffer[String]]

  def addKey(key: String, domain: String) = {
    val index = _keys.getOrElseUpdate(domain, new ArrayBuffer).length
    _map.getOrElseUpdate(domain, new mutable.HashMap)(key) = index
    _keys(domain) += key
    index
  }

  def apply(key: String, domain: String): Int = _map.getOrElse(domain, Map.empty[String, Int]).getOrElse(key, addKey(key, domain))
}

case class CachedID(id: String, domain: String)(implicit cache: Cache) extends ID {
  val idx: Int = cache(id, domain)
}