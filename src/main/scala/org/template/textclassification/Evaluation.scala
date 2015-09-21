package org.template.textclassification

import io.prediction.controller._



// 1. Create an accuracy metric for evaluating our supervised learning model.
case class Accuracy()
  extends AverageMetric[EmptyEvaluationInfo, Query, PredictedResult, ActualResult] {

  // Method for calculating prediction accuracy.
  def calculate(
    query: Query,
    predicted: PredictedResult,
    actual: ActualResult
  ) : Double = if (predicted.category == actual.category) 1.0 else 0.0
}



// 2. Define your evaluation object implementing the accuracy metric defined
// above.
object AccuracyEvaluation extends Evaluation {

  // Define Engine and Metric used in Evaluation.
  engineMetric = (
    TextClassificationEngine(),
    new Accuracy
  )
}



// 3. Set your engine parameters for evaluation procedure.
object EngineParamsList extends EngineParamsGenerator {

  // Set data source and preparator parameters.
  private[this] val baseEP = EngineParams(
    dataSourceParams = DataSourceParams(appName = "MyTextApp", evalK = Some(3)),
    preparatorParams = PreparatorParams(nGram = 2, 5000, true) 
  )

  // Set the algorithm params for which we will assess an accuracy score.
  engineParamsList = Seq(
    baseEP.copy(algorithmParamsList = Seq(("nb", NBAlgorithmParams(0.25)))),
    baseEP.copy(algorithmParamsList = Seq(("nb", NBAlgorithmParams(1.0)))),
    baseEP.copy(algorithmParamsList = Seq(("lr", LRAlgorithmParams(0.5)))),
    baseEP.copy(algorithmParamsList = Seq(("lr", LRAlgorithmParams(1.25))))
  )
}
