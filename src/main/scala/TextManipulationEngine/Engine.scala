package TextManipulationEngine

import io.prediction.controller._

class Query(
             val text: String
             ) extends Serializable

class PredictedResult (
                        val label: Double,
                        val confidence: Double
                        ) extends Serializable

class ActualResult(val label: Double) extends Serializable

object TextManipulationEngine extends EngineFactory {
  override
  def apply() = {
    new Engine(
      classOf[DataSource],
      classOf[Preparator],
      Map("sup" -> classOf[SupervisedAlgorithm]),
      classOf[Serving])
  }
}