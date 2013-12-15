package org.sameersingh.mf

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

case class SimpleID(id: String, idx: Int, domain: String) extends ID {
  override def toString = id
}

case class DoubleValue(double: Double) extends Val