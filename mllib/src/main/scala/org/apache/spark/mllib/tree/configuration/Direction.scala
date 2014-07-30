package org.apache.spark.mllib.tree.configuration

/**
 * :: Experimental ::
 * Enum to select the direction for a node
 */
object Direction extends Enumeration {
  type Direction = Value
  val Right, Left = Value
}
