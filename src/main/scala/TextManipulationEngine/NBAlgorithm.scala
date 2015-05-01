package TextManipulationEngine

import io.prediction.controller.{P2LAlgorithm, Params}
import org.apache.spark.SparkContext

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