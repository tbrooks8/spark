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

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.tree.configuration.Direction
import org.apache.spark.mllib.tree.configuration.Direction.Direction

import scala.annotation.tailrec
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class TreeSimplifier {
  case class Parent(parentIndex: Int, direction: Direction)

  protected[tree] def simplifyTree(
    tree: DecisionTreeModel,
    features: Vector,
    newTreeFeatures: Set[Int]): DecisionTreeModel = {
    val nodes = tree.nodes
    val parentStack: mutable.Stack[Parent] = mutable.Stack()
    val nodesRemainingStack: mutable.Stack[Int] = mutable.Stack()
    val newTree: ArrayBuffer[Node] = new ArrayBuffer()

    @tailrec
    def searchTree(node: Node): DecisionTreeModel = {
      if (nodesRemainingStack.isEmpty && node.isLeaf) {
        new DecisionTreeModel(newTree.toArray, 0, tree.algo)
      } else {
        if (node.isLeaf) {
          // This could throw if first node leaf
          setChild(newTree, parentStack)
          searchTree(nodes(nodesRemainingStack.pop()))
        } else if (newTreeFeatures contains node.split.get.feature) {
          if (parentStack.nonEmpty) {
            setChild(newTree, parentStack)
          }
          parentStack.push(Parent(newTree.length, Direction.Left), Parent(newTree.length, Direction.Right))
          newTree :+ new Node(node.id, node.predict, node.isLeaf, node.split, None, None, node.stats)
          nodesRemainingStack.push(node.leftNodeIndex.get, node.rightNodeIndex.get)
          searchTree(nodes(nodesRemainingStack.pop()))
        } else {
          searchTree(nodes(node.nextNodeIndex(features)))
        }
      }
    }

    searchTree(nodes(tree.rootNodeIndex))
  }

  private def setChild(newTree: ArrayBuffer[Node], parentStack: mutable.Stack[Parent])  {
    parentStack.pop() match {
      case Parent(index, Direction.Left) => newTree(index).leftNodeIndex = Some(newTree.size)
      case Parent(index, Direction.Right) => newTree(index).rightNodeIndex = Some(newTree.size)
    }


  }

}
