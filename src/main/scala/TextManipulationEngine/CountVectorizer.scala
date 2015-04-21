package TextManipulationEngine

import opennlp.tools.ngram.NGramModel
import opennlp.tools.tokenize.SimpleTokenizer
import opennlp.tools.util.StringList
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import scala.collection.mutable.{HashMap, LinkedHashSet}


// This class will take in as parameters an instance of type
// TrainingData, nMin, nMax which are the lower and upper bounds,
// respectively, of the model n-gram window. This class serves
// as an implementation of our data model.
class CountVectorizer (
                        val pd : PreparedData,
                        nMin : Int = 1,
                        nMax : Int = 2
                        ){


  // This private method will tokenize our text entries.
  private def tokenize (doc : String): Array[String] = {
    SimpleTokenizer.INSTANCE.tokenize(doc)
  }


  // This private method will help convert each text observation
  // and return a HashMap in which every token that appears in the
  // document is associated to the number of times it appears in the
  // document.
  private def hashData (doc : String) : HashMap[String, Double] = {

    val model = new NGramModel()
    model.add(new StringList(tokenize(doc): _*), nMin, nMax)
    val hashMap = new HashMap[String, Double]()
    for (x <- model.iterator)
      hashMap.put(x, model.getCount(x).toDouble)

    return hashMap
  }

  // Create token-gram universe.
  private val universe = new LinkedHashSet[String]()
  pd.data.map(
    e => hashData(e.text)
  ).foreach(
      e => e.keySet.foreach(universe.add)
    )

  // Transforms a given string document into a data vector
  // based on the given data model.
  def transform(doc : String) : Array[Double] = {

    val hashedData = hashData(doc)
    universe.toArray.map(
      e => hashedData.getOrElse(e, 0.0)
    )
  }

  // Returns a data instance that is ready to be used for
  // model training.
  def transformData : RDD[TransformedData] = {
    pd.data.map(
      e => TransformedData(
        (e.label, transform(e.text))
      ))
  }
}

// Class containing an individual data point
// transformed under a given data model.
case class TransformedData(
                          data : (Double, Array[Double])
                            )

