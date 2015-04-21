package TextManipulationEngine

import io.prediction.controller.{P2LAlgorithm, Params}
import org.apache.spark.SparkContext

case class SupervisedAlgorithmParams(
                                      lambda: Double,
                                      nMin: Int = 1,
                                      nMax: Int = 2
                                      ) extends Params


class SupervisedAlgorithm(params: SupervisedAlgorithmParams)
  extends P2LAlgorithm[PreparedData, SupervisedModel, Query, PredictedResult] {

  def train(sc: SparkContext, pd: PreparedData): SupervisedModel = {
    new SupervisedModel(pd, params.lambda, params.nMin, params.nMax)
  }

  def predict(model: SupervisedModel, query: Query): PredictedResult = {
    model.predict(query.text)
  }
}