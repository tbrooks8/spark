/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.tree.model

import org.apache.spark.annotation.Experimental
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.rdd.RDD

import scala.annotation.tailrec


@Experimental
class SimplifiableDecisionTreeModel(val nodes: Array[Node],val topNodeIdx: Int, val algo: Algo) extends Serializable {

  /**
   * Predict values for a single data point using the model trained.
   *
   * @param features array representing a single data point
   * @return Double prediction from the trained model
   */
  def predict(features: Vector): Double = {
    traverseTree(nodes(0), features)
  }

  /**
   * Predict values for the given data set using the model trained.
   *
   * @param features RDD representing data points to be predicted
   * @return RDD[Int] where each entry contains the corresponding prediction
   */
  def predict(features: RDD[Vector]): RDD[Double] = {
    features.map(x => predict(x))
  }

  @tailrec
  private def traverseTree(currentNode: Node, features: Vector): Double = {
    if (currentNode.isLeaf) {
      currentNode.predict
    } else {
      traverseTree(nodes(currentNode.nextNodeIndex(features)), features)
    }
  }
}
