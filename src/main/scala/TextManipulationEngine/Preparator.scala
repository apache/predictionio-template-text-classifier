package TextManipulationEngine

import io.prediction.controller.PPreparator
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD


class PreparedData (
                     val data : RDD[Observation]
                     ) extends Serializable


class Preparator extends PPreparator[TrainingData, PreparedData] {
  def prepare(sc : SparkContext, td: TrainingData): PreparedData = {
    new PreparedData(td.data)
  }
}
