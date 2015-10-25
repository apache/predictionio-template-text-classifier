package org.template.textclassification

import io.prediction.controller._



// 1. Define Query class which serves as a wrapper for
// new text data.
class Query(
  val text: String
) extends Serializable



// 2. Define PredictedResult class which serves as a
// wrapper for a predicted class label and the associated
// prediction confidence.
case class PredictedResult (
  val category: String,
  val confidence: Double
) extends Serializable





// 3. Define ActualResult class which serves as a wrapper
// for an observation's true class label.
class ActualResult(
  val category: String
) extends Serializable



// 4. Initialize the engine.
object TextClassificationEngine extends EngineFactory {
  override
  def apply() = {
    new Engine(
      classOf[DataSource],
      classOf[Preparator],
      Map(
        "VWlogisticSGD" -> classOf[VowpalLogisticRegressionWithSGDAlgorithm],
        "nb" -> classOf[NBAlgorithm],
        "lr" -> classOf[LRAlgorithm],
        "bid-lr" -> classOf[BIDMachLRAlgorithm]
      ), classOf[Serving]
    )
  }
}

