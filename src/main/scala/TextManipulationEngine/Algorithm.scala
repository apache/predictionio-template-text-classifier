package TextManipulationEngine

import io.prediction.controller.Params

case class AlgorithmParams(nCluster: Option[Int],
                           lambda: Option[Double],
                           nMin: Int = 1, nMax: Int = 2) extends Params


class Algorithm(ap: AlgorithmParams)