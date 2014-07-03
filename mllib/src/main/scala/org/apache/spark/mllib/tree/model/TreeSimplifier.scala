package org.apache.spark.mllib.tree.model

import org.apache.spark.mllib.linalg.Vector

import scala.annotation.tailrec
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
 * Created by timbrooks on 6/28/14.
 */
class TreeSimplifier {
  case class Parent(parentIndex: Int, direction: String)
  object Direction extends Enumeration {
    type Direction = Value
    val Right, Left = Value
  }

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
          handleLeaf(newTree, parentStack)
          searchTree(nodes(node.nextNodeIndex(features)))
        } else if (newTreeFeatures contains node.split.get.feature) {
          searchTree(nodes(node.nextNodeIndex(features)))
        } else {
          searchTree(nodes(node.nextNodeIndex(features)))
        }
      }
    }

    searchTree(nodes(tree.rootNodeIndex))
  }

  private def handleLeaf(newTree: ArrayBuffer[Node], parentStack: mutable.Stack[Parent])  {
    parentStack.pop() match {
      case Parent(index, Left) => newTree(index).leftNodeIndex = Some(newTree.size)
      case Parent(index, Right) => newTree(index).leftNodeIndex = Some(newTree.size)
    }


  }

}
