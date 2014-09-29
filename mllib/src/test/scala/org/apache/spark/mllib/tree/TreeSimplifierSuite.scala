package org.apache.spark.mllib.tree

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.impurity.Gini
import org.apache.spark.mllib.tree.model.TreeSimplifier
import org.apache.spark.mllib.util.LocalSparkContext
import org.scalatest.FunSuite

/**
 * Created by timbrooks on 7/30/14.
 */
class TreeSimplifierSuite extends FunSuite with LocalSparkContext {

  test("Test tree is simplified to defined features") {
    val strategy = new Strategy(algo = Classification, impurity = Gini, maxDepth = 5,
      numClassesForClassification = 3)
    val input = sc.parallelize(generateContinuousDataPointsForMulticlass())

    val tree = DecisionTree.train(input, strategy)
    val simplifier: TreeSimplifier = new TreeSimplifier
    val features: Vector = Vectors.dense(Array(1.0, 2.0, 3.0))

    for (n <- tree.nodes) {println(n)}

    simplifier.simplifyTree(tree, features, Set(1))
  }

  def generateCategoricalDataPointsForMulticlass(): Array[LabeledPoint] = {
    val arr = new Array[LabeledPoint](3000)
    for (i <- 0 until 3000) {
      if (i < 1000) {
        arr(i) = new LabeledPoint(2.0, Vectors.dense(2.0, 2.0))
      } else if (i < 2000) {
        arr(i) = new LabeledPoint(1.0, Vectors.dense(1.0, 2.0))
      } else {
        arr(i) = new LabeledPoint(2.0, Vectors.dense(2.0, 2.0))
      }
    }
    arr
  }

  def generateContinuousDataPointsForMulticlass(): Array[LabeledPoint] = {
    val arr = new Array[LabeledPoint](3000)
    for (i <- 0 until 3000) {
      if (i < 2000) {
        arr(i) = new LabeledPoint(2.0, Vectors.dense(2.0, i))
      } else {
        arr(i) = new LabeledPoint(1.0, Vectors.dense(2.0, i))
      }
    }
    arr
  }

}
