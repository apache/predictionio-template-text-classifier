package TextManipulationEngine

import io.prediction.controller.SanityCheck
import io.prediction.controller.{PPreparator, Params}
import io.prediction.data.store.LEventStore
import io.prediction.data.store.PEventStore
import org.apache.spark.SparkContext


case class PreparatorParams(
                             nMin: Int,
                             nMax: Int,
                             tfidf: Boolean
                             ) extends Params


class Preparator(pp: PreparatorParams) extends PPreparator[TrainingData, PreparedData] {
  def prepare(sc : SparkContext, td: TrainingData): PreparedData = {




    new PreparedData(new DataModel(td, pp.nMin, pp.nMax, pp.tfidf))
  }
}

class PreparedData(
                    val dataModel: DataModel
                    ) extends Serializable


