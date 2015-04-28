package TextManipulationEngine

import grizzled.slf4j.Logger
import io.prediction.controller.EmptyEvaluationInfo
import io.prediction.controller.PDataSource
import io.prediction.controller.Params
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

  private def readStopWords(sc : SparkContext) : Set[String] = {
    PEventStore.find(
      appName = dsp.appName,
      entityType = Some("resource"),
      eventNames = Some(List("stopwords"))
    )(sc).map(e =>
      e.properties.get[String]("word")
      ).collect.toSet
  }


  //
  override
  def readTraining(sc: SparkContext): TrainingData = {
    new TrainingData(readEventData(sc), readStopWords(sc))
  }

  override
  def readEval(sc: SparkContext):
  Seq[(TrainingData, EmptyEvaluationInfo, RDD[(Query, ActualResult)])] = {
    val data = readEventData(sc).zipWithIndex()

    (0 until dsp.evalK.get).map(
      k => (new TrainingData(data.filter(_._2 % dsp.evalK.get != k).map(_._1), readStopWords((sc))),
        new EmptyEvaluationInfo(),
        data.filter(_._2 % dsp.evalK.get == k).map(_._1).map(e => (new Query(e.text), new ActualResult(e.label)))
        )
    )
  }
}


case class Observation(
                        label : Double,
                        text : String
                        ) extends Serializable


class TrainingData(
                    val data : RDD[Observation],
                    val stopWords : Set[String]
                    ) extends Serializable
