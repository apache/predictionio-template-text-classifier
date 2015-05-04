package TextManipulationEngine

import io.prediction.controller.LServing

// 1. Define serving component.
class Serving extends LServing[Query, PredictedResult] {



  override
  def serve(query: Query, predictedResults: Seq[PredictedResult]):
  PredictedResult = predictedResults.maxBy(e => e.confidence)
}


