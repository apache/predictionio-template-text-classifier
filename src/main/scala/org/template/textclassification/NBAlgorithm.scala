package org.template.textclassification

import io.prediction.controller.{P2LAlgorithm, Params}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.classification.NaiveBayesModel

import scala.math._

// 1. Define parameters for Supervised Learning Model. We are
// using a Naive Bayes classifier, which gives us only one
// hyperparameter in this stage.

case class NBAlgorithmParams(
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
    pd.transformData, lambda)



  // 2. Set up framework for performing the required Matrix
  // Multiplication for the prediction rule explained in the
  // tutorial.

  private def innerProduct (x : Array[Double], y : Array[Double]) : Double = {
    require(x.length == y.length)

    x.zip(y).map(e => e._1 * e._2).sum
  }



  // 3. Given a document string, return a vector of corresponding
  // class membership probabilities.

  private def getScores(doc: String): Array[Double] = {
    // Helper function used to normalize probability scores.
    // Returns an object of type Array[Double]
    val normalize = (u: Array[Double]) => u.map(_ / u.sum)
    // Vectorize query,
    val x: Array[Double] = pd.transform(doc).toArray

    normalize(
      nb.pi
      .zip(nb.theta)
      .map(
      e => exp(innerProduct(e._2, x) + e._1))
    )
  }

  // 4. Implement predict method for our model using
  // the prediction rule given in tutorial.

  def predict(doc : String) : PredictedResult = {
    try {
      val x: Array[Double] = getScores(doc)
      val y: (Double, Double) = (nb.labels zip x).maxBy(_._2)
      new PredictedResult(pd.categoryMap.getOrElse(y._1, ""), y._2)
    } catch {
      case e : IllegalArgumentException => PredictedResult(pd.majorityCategory, 0)
    }
  }
}