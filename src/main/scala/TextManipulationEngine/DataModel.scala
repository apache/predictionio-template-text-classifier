package TextManipulationEngine

import opennlp.tools.ngram.NGramModel
import opennlp.tools.tokenize.SimpleTokenizer
import opennlp.tools.util.StringList
import org.apache.spark.rdd.RDD

import scala.collection.JavaConversions._


// This class will take in as parameters an instance of type
// TrainingData, nMin, nMax which are the lower and upper bounds,
// respectively, of the model n-gram window. This class serves
// as an implementation of our data model.

class DataModel(
                 td: TrainingData,
                 nMin: Int,
                 nMax: Int
                 ) {


  // This private method will tokenize our text entries.
  private def tokenize (doc : String): Array[String] = {
    SimpleTokenizer.INSTANCE.tokenize(doc)
  }


  // This private method will help convert each text observation
  // and return a HashMap in which every token that appears in the
  // document is associated to the number of times it appears in the
  // document.
  private def hashDoc(doc: String): Map[String, Double] = {

    val model = new NGramModel()
    model.add(new StringList(tokenize(doc): _*), nMin, nMax)
    model.iterator.map(
      x => (x.toString, model.getCount(x).toDouble)
    ).toMap
  }


  // Create token-gram universe.
  private def createUniverse(u: RDD[Map[String, Double]]): Array[String] = {
    u.map(e => e.keySet).reduce((a, b) => a.union(b)).toArray
  }

  private val universe = createUniverse(
    td.data.map(
      e => hashDoc(e.text)
    ))

  // Transforms a given string document into a data vector
  // based on the given data model.
  def transform(doc : String) : Array[Double] = {

    val hashedDoc = hashDoc(doc)
    universe.map(
      e => hashedDoc.getOrElse(e, 0.0)
    )
  }

  // Returns a data instance that is ready to be used for
  // model training.
  def transformData: RDD[(Double, Array[Double])] = {
    td.data.map(
      e => (e.label, transform(e.text))
    )
  }
}