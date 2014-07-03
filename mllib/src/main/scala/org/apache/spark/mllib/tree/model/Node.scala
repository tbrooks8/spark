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

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.Logging
import org.apache.spark.mllib.tree.configuration.FeatureType._
import org.apache.spark.mllib.linalg.Vector

/**
 * :: DeveloperApi ::
 * Node in a decision tree
 * @param id integer node id
 * @param predict predicted value at the node
 * @param isLeaf whether the leaf is a node
 * @param split split to calculate left and right nodes
 * @param leftNodeIndex  left child index
 * @param rightNodeIndex right child index
 * @param stats information gain stats
 */
@DeveloperApi
class Node (
    val id: Int,
    val predict: Double,
    val isLeaf: Boolean,
    val split: Option[Split],
    var leftNodeIndex: Option[Int],
    var rightNodeIndex: Option[Int],
    val stats: Option[InformationGainStats]) extends Serializable with Logging {

  override def toString = "id = " + id + ", isLeaf = " + isLeaf + ", predict = " + predict + ", " +
    "split = " + split + ", stats = " + stats

  /**
   * build the left node and right nodes if not leaf
   * @param nodes array of nodes
   */
  def build(nodes: Array[Node]): Unit = {

    logDebug("building node " + id + " at level " +
      (scala.math.log(id + 1)/scala.math.log(2)).toInt )
    logDebug("id = " + id + ", split = " + split)
    logDebug("stats = " + stats)
    logDebug("predict = " + predict)
    if (!isLeaf) {
      leftNodeIndex = Some(id * 2 + 1)
      rightNodeIndex = Some(id * 2 + 2)
      nodes(leftNodeIndex.get).build(nodes)
      nodes(rightNodeIndex.get).build(nodes)
    }
  }

  /**
   * solve for next node index
   * @param feature feature value
   * @return predicted value
   */
  def nextNodeIndex(feature: Vector): Int = {
    if (split.get.featureType == Continuous) {
      if (feature(split.get.feature) <= split.get.threshold) {
        leftNodeIndex.get
      } else {
        rightNodeIndex.get
      }
    } else {
      if (split.get.categories.contains(feature(split.get.feature))) {
        leftNodeIndex.get
      } else {
        rightNodeIndex.get
      }
    }
  }
}
