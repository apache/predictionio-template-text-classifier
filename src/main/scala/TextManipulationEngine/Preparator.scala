package TextManipulationEngine

import io.prediction.controller.{PPreparator, Params}
import io.prediction.data.store.LEventStore
import io.prediction.data.store.PEventStore
import org.apache.spark.SparkContext


case class PreparatorParams(
                             nMin: Int,
                             nMax: Int,
                             tfidf: Boolean,
                             appName : String
                             ) extends Params


class Preparator(pp: PreparatorParams) extends PPreparator[TrainingData, PreparedData] {
  def prepare(sc : SparkContext, td: TrainingData): PreparedData = {

    val stopWords = PEventStore.find(
      appName = pp.appName,
      entityType = Some("stopword"),
      eventNames = Some(List("stopwords"))
    )(sc).map(e =>
        e.properties.get[String]("word")
      ).collect.toSet


    new PreparedData(new DataModel(td, pp.nMin, pp.nMax, pp.tfidf, stopWords))
  }
}

class PreparedData(
                    val dataModel: DataModel
                    ) extends Serializable {
  override def toString() : String = {
    dataModel.td.data.count.toString +
      dataModel.stopWords.size.toString
  }
}

