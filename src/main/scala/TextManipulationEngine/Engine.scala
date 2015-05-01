package TextManipulationEngine

import io.prediction.controller._



// 1. Define Query class which serves as a wrapper for
// new text data.
class Query(
  val text: String
) extends Serializable



// 2. Define PredictedResult class which serves as a
// wrapper for a predicted class label and the associated
// prediction confidence.
class PredictedResult (
  val label: Double,
  val confidence: Double
) extends Serializable



// 3. Define ActualResult class which serves as a wrapper
// for an observation's true class label.
class ActualResult(
  val label: Double
) extends Serializable



// 4. Initialize the engine.
object TextManipulationEngine extends EngineFactory {
  override
  def apply() = {
    new Engine(
      classOf[DataSource],
      classOf[Preparator],
      Map(
        "nb" -> classOf[NBAlgorithm]
      ), classOf[Serving]
    )
  }
}