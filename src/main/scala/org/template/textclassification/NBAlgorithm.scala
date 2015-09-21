package org.template.textclassification

import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.classification.NaiveBayesModel
import org.apache.spark.mllib.linalg.Vector
import com.github.fommil.netlib.F2jBLAS

import scala.math._

// 1. Define parameters for Supervised Learning Model. We are
// using a Naive Bayes classifier, which gives us only one
// hyperparameter in this stage.

case class  NBAlgorithmParams(
  lambda: Double
) extends Params



// 2. Define SupervisedAlgorithm class.

class NBAlgorithm(
  val sap: NBAlgorithmParams
) extends P2LAlgorithm[PreparedData, NBModel, Query, PredictedResult] {

  // Train your model.
  def train(sc: SparkContext, pd: PreparedData): NBModel = {
    new NBModel(pd, sap.lambda)
  }

  // Prediction method for trained model.
  def predict(model: NBModel, query: Query): PredictedResult = {
    model.predict(query.text)
  }
}

class NBModel(
val pd: PreparedData,
lambda: Double
) extends Serializable {



  // 1. Fit a Naive Bayes model using the prepared data.

  private val nb : NaiveBayesModel = NaiveBayes.train(
    pd.transformedData.map(x=>x.point), lambda)



  // 2. Set up linear algebra framework.

  private def innerProduct (x : Array[Double], y : Array[Double]) : Double = {
    x.zip(y).map(e => e._1 * e._2).sum
  }

  val normalize = (u: Array[Double]) => {
    val uSum = u.sum

    u.map(e => e / uSum)
  }



  private val scoreArray = nb.pi.zip(nb.theta)

  // 3. Given a document string, return a vector of corresponding
  // class membership probabilities.

  private def getScores(doc: String): Array[Double] = {
    // Helper function used to normalize probability scores.
    // Returns an object of type Array[Double]

    // Vectorize query,
    val x: Vector = pd.transform(doc).vector

    val z = scoreArray
      .map(e => innerProduct(e._2, x.toArray) + e._1)

    normalize((0 until z.size).map(k => exp(z(k) - z.max)).toArray)
  }

  // 4. Implement predict method for our model using
  // the prediction rule given in tutorial.

  def predict(doc : String) : PredictedResult = {
    val x: Array[Double] = getScores(doc)
    val y: (Double, Double) = (nb.labels zip x).maxBy(_._2)
    new PredictedResult(pd.categoryMap.getOrElse(y._1, ""), y._2)
  }
}