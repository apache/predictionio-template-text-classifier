package TextManipulationEngine

import io.prediction.controller.{P2LAlgorithm, Params}
import org.apache.spark.SparkContext

// 1. Define parameters for Supervised Learning Model. We are
// using a Naive Bayes classifier, which gives us only one
// hyperparameter in this stage.

case class SupervisedAlgorithmParams(
  lambda: Double
) extends Params



// 2. Define SupervisedAlgorithm class.

class SupervisedAlgorithm(
  val sap: SupervisedAlgorithmParams
) extends P2LAlgorithm[PreparedData, SupervisedModel, Query, PredictedResult] {

  // Train your model.
  def train(sc: SparkContext, pd: PreparedData): SupervisedModel = {
    new SupervisedModel(pd, sap.lambda)
  }

  // Prediction method for trained model.
  def predict(model: SupervisedModel, query: Query): PredictedResult = {
    model.predict(query.text)
  }
}