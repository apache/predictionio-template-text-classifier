package org.template.textclassification

import io.prediction.controller.Params
import io.prediction.controller.P2LAlgorithm
import io.prediction.workflow.FakeRun
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.SQLContext

import scala.math._


case class LRAlgorithmParams (
  regParam  : Double
) extends Params


class LRAlgorithm(
  val sap: LRAlgorithmParams
) extends P2LAlgorithm[PreparedData, LRModel, Query, PredictedResult] {

  // Train your model.
  def train(sc: SparkContext, pd: PreparedData): LRModel = {
    new LRModel(sc, pd, sap.regParam)
  }

  // Prediction method for trained model.
  def predict(model: LRModel, query: Query): PredictedResult = {
    model.predict(query.text)
  }
}

class LRModel (
  sc : SparkContext,
  pd : PreparedData,
  regParam : Double
) extends Serializable {

  private val sql : SQLContext = new SQLContext(sc)
  import sql.implicits._

  private val binaryDFs : Seq[(Double, DataFrame)]= pd.categoryMap.keys.toSeq.map(
    lab => {
      val labData = pd.transformedData.map(e => LabeledPoint(
        if (e.label == lab) 1.0 else 0.0,
        e.features
      ))

      (lab, labData.toDF)
    }
  )

  private val lr = new LogisticRegression()
    .setMaxIter(10000)
    .setThreshold(0.5)
    .setRegParam(regParam)

  private val lrModels : Seq[(Double, Array[Double], Double)] = binaryDFs.map(
      e => {
        val fit = lr.fit(e._2)
        (e._1, fit.weights.toArray, fit.intercept)
      }
  )

  private val normalize = (u: Array[Double]) => u.map(_ / u.sum)

  private def innerProduct (x : Array[Double], y : Array[Double]) : Double = {
    require(x.length == y.length)

    x.zip(y).map(e => e._1 * e._2).sum
  }

  def predict(text : String): PredictedResult = {
    val x : Array[Double] = pd.transform(text).toArray

    val pred = lrModels.map(
      e => {
        val z = exp(innerProduct(e._2, x) + e._3)

        (e._1, 1 / (1 + z))
      }
    ).maxBy(_._2)

    PredictedResult(pd.categoryMap(pred._1), pred._2)
  }
}

