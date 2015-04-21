package TextManipulationEngine

import io.prediction.controller.{P2LAlgorithm, Params}
import org.apache.spark.SparkContext

case class AlgorithmParams(nCluster: Option[Int],
                           lambda: Option[Double],
                           nMin: Int = 1,
                           nMax: Int = 2) extends Params {
  require(nCluster.isDefined || lambda.isDefined)
}


class Algorithm(ap: AlgorithmParams)
  extends P2LAlgorithm[PreparedData, Model, Query, PredictedResult] {

  def train(sc: SparkContext, pd: PreparedData): Model = {
    if (ap.lambda.isDefined) {
      new SupervisedModel(pd, ap.lambda.get)
    } else {
      new UnsupervisedModel(pd, ap.nCluster.get)
    }
  }

  def predict(model: Model, query: Query): PredictedResult = {
    model.predict(query.text)
  }
}