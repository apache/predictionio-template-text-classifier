package TextManipulationEngine


import io.prediction.controller.PPreparator
import io.prediction.controller.Params
import org.apache.spark.SparkContext



// 1. Initialize Preparator parameters. Recall that for our data
// representation we are only required to input the n-gram window
// components.

case class PreparatorParams(
  nMin: Int,
  nMax: Int
) extends Params



// 2. Initialize your Preparator class.

class Preparator(pp: PreparatorParams) extends PPreparator[TrainingData, PreparedData] {

  // Prepare your training data.
  def prepare(sc : SparkContext, td: TrainingData): PreparedData = {
    new PreparedData(new DataModel(td, pp.nMin, pp.nMax))
  }
}


// Define PreparedData, a wrapper for our data model.
class PreparedData(
  val dataModel: DataModel
) extends Serializable


