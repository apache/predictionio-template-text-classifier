package TextManipulationEngine

import grizzled.slf4j.Logger
import io.prediction.controller.{EmptyEvaluationInfo,
  EmptyActualResult ,
  Params,
  PDataSource}
import io.prediction.data.storage.Storage
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD


case class DataSourceParams(appId : Int) extends Params

class DataSource (val dsp : DataSourceParams)
  extends PDataSource[TrainingData, EmptyEvaluationInfo, Query, EmptyActualResult] {

  @transient lazy val logger = Logger[this.type]

  // Helper function used to store data given sc,
  // a SparkContext.
  private def readEventData(sc: SparkContext) : RDD[Observation] = {
    // Get PEvents database instance.
    val eventsDb = Storage.getPEvents()

    // Store observations from event server.
    eventsDb.find(
      appId = dsp.appId,
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
}



case class Observation(
                        label : Double,
                        text : String
                        ) extends Serializable


class TrainingData(
                    val data : RDD[Observation]
                    ) extends Serializable
