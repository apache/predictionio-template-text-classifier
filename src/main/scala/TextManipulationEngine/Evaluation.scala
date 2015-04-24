package TextManipulationEngine


import io.prediction.controller._

import scala.math.pow

case class MeanSquaredError() extends
AverageMetric[EmptyEvaluationInfo, Query, PredictedResult, ActualResult] {
  def calculate(query: Query, predicted: PredictedResult, actual: ActualResult)
  : Double = pow(predicted.label - actual.label, 2)
}


object MSEEvaluation extends Evaluation {
  // Define Engine and Metric used in Evaluation
  override val engineEvaluator = (TextManipulationEngine(),
    MetricEvaluator(
      metric = MeanSquaredError()
    ))
}

object EngineParamsList extends EngineParamsGenerator {
  // Define list of EngineParams used in Evaluation

  // First, we define the base engine params. It specifies the appId from which
  // the data is read, and a evalK parameter is used to define the
  // cross-validation.
  private[this] val baseEP = EngineParams(
    dataSourceParams = DataSourceParams(appName = "test", evalK = Some(5)))

  // Second, we specify the engine params list by explicitly listing all
  // algorithm parameters. In this case, we evaluate 3 engine params, each with
  // a different algorithm params value.

  // In this example we will primarily focus on the appropriate value of the additive smoothing constant,
  // and leave the number of n-grams fixed. The number of n-grams itself is a model hyperparameter
  // and should also be tuned.
  engineParamsList = Seq(
    baseEP.copy(algorithmParamsList = Seq(("algo", SupervisedAlgorithmParams(0.3)))),
    baseEP.copy(algorithmParamsList = Seq(("algo", SupervisedAlgorithmParams(0.5)))),
    baseEP.copy(algorithmParamsList = Seq(("algo", SupervisedAlgorithmParams(1)))))
}