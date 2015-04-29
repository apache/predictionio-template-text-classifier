package TextManipulationEngine

import io.prediction.controller.SanityCheck
import io.prediction.controller.{PPreparator, Params}
import io.prediction.data.store.LEventStore
import io.prediction.data.store.PEventStore
import org.apache.spark.SparkContext


case class PreparatorParams(
                             nMin: Int,
                             nMax: Int
                             ) extends Params


class Preparator(pp: PreparatorParams) extends PPreparator[TrainingData, PreparedData] {
  def prepare(sc : SparkContext, td: TrainingData): PreparedData = {




    new PreparedData(new DataModel(td, pp.nMin, pp.nMax))
  }
}

class PreparedData(
                    val dataModel: DataModel
                    ) extends Serializable


