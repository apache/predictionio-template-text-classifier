package TextManipulationEngine

import io.prediction.controller.{P2LAlgorithm, Params}
import org.apache.spark.SparkContext

case class SupervisedAlgorithmParams(
                                      lambda: Double
                                      ) extends Params


class SupervisedAlgorithm(sap: SupervisedAlgorithmParams)
  extends P2LAlgorithm[PreparedData, SupervisedModel, Query, PredictedResult] {

  def train(sc: SparkContext, pd: PreparedData): SupervisedModel = {
    new SupervisedModel(pd, sap.lambda)
  }

  def predict(model: SupervisedModel, query: Query): PredictedResult = {
    model.predict(query.text)
  }
}