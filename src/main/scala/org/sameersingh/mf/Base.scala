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