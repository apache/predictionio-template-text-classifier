package org.template.textclassification

import java.io._

import BIDMat.{DMat, Mat}
import io.prediction.controller.Params
import io.prediction.controller.P2LAlgorithm
import io.prediction.workflow.FakeRun
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.UserDefinedFunction
import com.github.fommil.netlib.F2jBLAS
import org.template.textclassification.NativeLRModel


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
) extends Serializable with NativeLRModel {
  private val labels: Seq[Double] = pd.categoryMap.keys.toSeq

  val data = prepareDataFrame(sc, pd, labels)

  private val lrModels = fitLRModels

  def fitLRModels:Seq[(Double, LREstimate)] = {
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setThreshold(0.5)
      .setRegParam(regParam)

    // 3. Create a logistic regression model for each class.
    val lrModels: Seq[(Double, LREstimate)] = labels.map(
      label => {
        val lab = label.toInt.toString

        //val (categories, features) = getDMatsFromData(lab)


        val fit = lr.setLabelCol(lab).fit(
          data.select(lab, "features")
        )


        // Return (label, feature coefficients, and intercept term.
        (label, LREstimate(fit.weights.toArray, fit.intercept))

      }
    )
    lrModels
  }

  def predict(text : String): PredictedResult = {
    predict(text, pd, lrModels)
  }


}


