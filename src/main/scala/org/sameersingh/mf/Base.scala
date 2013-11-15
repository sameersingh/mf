package org.sameersingh.mf

/**
 * Created with IntelliJ IDEA.
 * User: sameer
 * Date: 11/14/13
 * Time: 2:20 PM
 * To change this template use File | Settings | File Templates.
 */
trait ID {
  def id: String
  def idx: Int
  def apply = id
}

trait Val {
  def double: Double
  def apply = double
}