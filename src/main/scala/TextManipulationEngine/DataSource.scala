package TextManipulationEngine

import grizzled.slf4j.Logger
import io.prediction.controller.{EmptyEvaluationInfo, PDataSource, Params}
import io.prediction.data.store.PEventStore
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD


case class DataSourceParams(
                             appName: String,
                             evalK: Option[Int]
                             ) extends Params

class DataSource (val dsp : DataSourceParams)
  extends PDataSource[TrainingData, EmptyEvaluationInfo, Query, ActualResult] {

  @transient lazy val logger = Logger[this.type]

  // Helper function used to store data given sc,
  // a SparkContext.
  private def readEventData(sc: SparkContext) : RDD[Observation] = {
    // Get PEvents database instance.
    PEventStore.find(
      appName = dsp.appName,
      entityType = Some("source"),
      eventNames = Some(List("documents"))

      // Convert collected RDD of events to and RDD of Observation
      // objects.
    )(sc).map(e => Observation(
      e.properties.get[Double]("label"),
      e.properties.get[String]("text")
    ))
  }

  //
  override
  def readTraining(sc: SparkContext): TrainingData = {
    new TrainingData(readEventData(sc))
  }

  override
  def readEval(sc: SparkContext):
  Seq[(TrainingData, EmptyEvaluationInfo, RDD[(Query, ActualResult)])] = {
    val data = readEventData(sc).zipWithIndex

    (0 until dsp.evalK.get).map(
      k => (new TrainingData(data.filter(_._2 % k != 0).map(_._1)),
        new EmptyEvaluationInfo,
        data.filter(_._2 % k == 0).map(_._1).map(e => (new Query(e.text), new ActualResult(e.label)))
        )
    )
  }
}



case class Observation(
                        label : Double,
                        text : String
                        ) extends Serializable


class TrainingData(
                    val data : RDD[Observation]
                    ) extends Serializable
