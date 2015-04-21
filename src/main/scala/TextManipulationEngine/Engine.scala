package TextManipulationEngine

import io.prediction.controller._

import scala.collection.immutable

class Query(
             val text: String
             ) extends Serializable

class PredictedResult (
                      label : Double,
                      val confidence : Double
                        ) extends Serializable {
  private val categories: immutable.HashMap[Double, String] = immutable.HashMap(
    0.0 -> "alt.atheism",
    1.0 -> "comp.graphics",
    2.0 -> "comp.os.ms-windows.misc",
    3.0 -> "comp.sys.ibm.pc.hardware",
    4.0 -> "comp.sys.mac.hardware",
    5.0 -> "comp.windows.x",
    6.0 -> "misc.forsale",
    7.0 -> "rec.autos",
    8.0 -> "rec.motorcycles",
    9.0 -> "rec.sport.baseball",
    10.0 -> "rec.sport.hockey",
    11.0 -> "sci.crypt",
    12.0 -> "sci.electronics",
    13.0 -> "sci.med",
    14.0 -> "sci.space",
    15.0 -> "soc.religion.christian",
    16.0 -> "talk.politics.guns",
    17.0 -> "talk.politics.mideast",
    18.0 -> "talk.politics.misc",
    19.0 -> "talk.religion.misc"
  )

  val category = categories.get(label)
}

object TextManipulationEngine extends IEngineFactory {
  override
  def apply() = {
    new Engine(
      classOf[DataSource],
      classOf[Preparator],
      Map("algo" -> classOf[Algorithm]),
      classOf[Serving])
  }
}