package TextManipulationEngine

import io.prediction.controller.{PPreparator, Params}
import org.apache.spark.SparkContext


case class PreparatorParams(
                             nMin: Int,
                             nMax: Int,
                             tfidf: Boolean
                             ) extends Params


class Preparator(pp: PreparatorParams) extends PPreparator[TrainingData, PreparedData] {
  def prepare(sc : SparkContext, td: TrainingData): PreparedData = {

    val stopWords = sc.textFile(
      "./data/common-english-words.txt"
    ).map(
        line => line.split(",").toSet
      ).first

    new PreparedData(new DataModel(td, pp.nMin, pp.nMax, pp.tfidf, stopWords))
  }
}

class PreparedData(
                    val dataModel: DataModel
                    ) extends Serializable