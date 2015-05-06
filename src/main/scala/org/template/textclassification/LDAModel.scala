package org.template.textclassification

import org.apache.spark.mllib.clustering.LDA
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.math.log


class LDAModel(
  val pd : PreparedData,
  nClust : Int,
  lambda: Double
) extends Serializable {


  private val dataRDD = pd.dataModel.transformData.cache

  private def createLDAMatrix : Matrix = {
    val ldaMatrix: Matrix = new LDA().setK(nClust).run(
      dataRDD
      .map(e => e.features)
      .zipWithIndex
      .map(_.swap)
      .cache
    ).topicsMatrix

    Matrices.dense(
      ldaMatrix.numRows,
      ldaMatrix.numCols,
      ldaMatrix.toArray.map(log)
    ).transpose
  }


  private val ldaMatrix : Matrix = createLDAMatrix


  // RDD[(cluster assignment, data point)]
  private val cluster : RDD[(Int, LabeledPoint)]= dataRDD.map(
    e => (
      ldaMatrix.multiply(new DenseVector(e.features.toArray))
        .toArray
        .zipWithIndex
        .maxBy(_._1)._2,
      e
    )
  )

  private val dataClusters : Array[RDD[LabeledPoint]]= cluster.keys.collect.map(
    k => cluster.filter(e => e._1 == k).values
  )






}
