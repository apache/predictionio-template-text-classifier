package TextManipulationEngine


import io.prediction.controller._

case class Accuracy()
  extends AverageMetric[EmptyEvaluationInfo, Query, PredictedResult, ActualResult] {
  def calculate(query: Query, predicted: PredictedResult, actual: ActualResult)
  : Double = if (predicted.label == actual.label) 1.0 else 0.0
}


object AccuracyEvaluation extends Evaluation {
  // Define Engine and Metric used in Evaluation
  engineMetric = (
    TextManipulationEngine(),
    new Accuracy())
}

object EngineParamsList extends EngineParamsGenerator {
  // Define list of EngineParams used in Evaluation

  // First, we define the base engine params. It specifies the appId from which
  // the data is read, and a evalK parameter is used to define the
  // cross-validation.
  private[this] val baseEP = EngineParams(
    dataSourceParams = DataSourceParams(appName = "marco-testapp", evalK = Some(5)),
    preparatorParams = PreparatorParams(1, 2, tfidf = true))

  // Second, we specify the engine params list by explicitly listing all
  // algorithm parameters. In this case, we evaluate 3 engine params, each with
  // a different algorithm params value.

  // In this example we will primarily focus on the appropriate value of the additive smoothing constant,
  // and leave the number of n-grams fixed. The number of n-grams itself is a model hyperparameter
  // and should also be tuned.
  engineParamsList = Seq(
    baseEP.copy(algorithmParamsList = Seq(("sup", SupervisedAlgorithmParams(0.5)))))
}