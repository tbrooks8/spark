package org.apache.spark.mllib.tree.model

import org.apache.spark.mllib.linalg.Vector

/**
 * Created by timbrooks on 6/28/14.
 */
class TreeSimplifier {

  protected[tree] def simplifyTree(
    tree: DecisionTreeModel,
    features: Vector,
    newTreeFeatures: Array[Int]): DecisionTreeModel = {
    tree
  }

}
