package org.template.textclassification


import io.prediction.controller.PPreparator
import io.prediction.controller.Params
import opennlp.tools.ngram.NGramModel
import opennlp.tools.tokenize.SimpleTokenizer
import opennlp.tools.util.StringList
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.collection.immutable.HashMap
import scala.collection.JavaConversions._
import scala.math._


// 1. Initialize Preparator parameters. Recall that for our data
// representation we are only required to input the n-gram window
// components.

case class PreparatorParams(
  nMin: Int,
  nMax: Int,
  inverseIdfMin : Double,
  inverseIdfMax : Double
) extends Params



// 2. Initialize your Preparator class.

class Preparator(pp: PreparatorParams) extends PPreparator[TrainingData, PreparedData] {

  // Prepare your training data.
  def prepare(sc : SparkContext, td: TrainingData): PreparedData = {
    new PreparedData(td, pp.nMin, pp.nMax, pp.inverseIdfMin, pp. inverseIdfMax)
  }
}

//------PreparedData------------------------

class PreparedData (
val td : TrainingData,
val nMin: Int,
val nMax: Int,
val inverseIdfMin : Double,
val inverseIdfMax : Double
) extends Serializable {


  // 1. Tokenizer: document => token list.
  // Takes an individual document and converts it to
  // a list of allowable tokens.

  private def tokenize (doc : String): Array[String] = {
    SimpleTokenizer.INSTANCE
    .tokenize(doc.toLowerCase)
    .filter(e => ! td.stopWords.contains(e))
  }


  // 2. Hasher: Array[tokens] => Map(n-gram -> n-gram document tf).

  private def hash (tokenList : Array[String]): HashMap[String, Double] = {
    // Initialize an NGramModel from OpenNLP tools library,
    // and add the list of allowable tokens to the n-gram model.
    val model : NGramModel = new NGramModel()
    model.add(new StringList(tokenList: _*), nMin, nMax)

    val map : HashMap[String, Double] = HashMap(
      model.iterator.map(
        x => (x.toString, model.getCount(x).toDouble)
      ).toSeq : _*
    )

    val mapSum = map.values.sum

    // Divide by the total number of n-grams in the document
    // to obtain n-gram frequency.
    map.map(e => (e._1, e._2 / mapSum))

  }


  // 3. Bigram universe extractor: RDD[bigram hashmap] => RDD[(n-gram, n-gram idf)]

  private def createUniverse(u: RDD[HashMap[String, Double]]): RDD[(String, Double)] = {
    // Total number of documents (should be 11314).
    val numDocs: Double = td.data.count.toDouble
    u.flatMap(e => e.map(f => (f._1, 1.0)))
    .reduceByKey(_ + _)
    .filter(e => {
      val docFreq = e._2 / numDocs

      // Cut out n-grams with inverse i.d.f. greater/less than or equal to min/max
      // cutoff.
      docFreq >= inverseIdfMin && docFreq <= inverseIdfMax
    })
    .map(e => (e._1, log(numDocs / e._2)))
  }


  // 4. Set private class variables for use in data transformations.

  // Create ngram to idf hashmap for every n-gram in universe:
  //    Map(n-gram -> n-gram idf)
  private val idf : HashMap[String, Double] = HashMap(
    createUniverse(
      td.data
      .map(e => hash(tokenize(e.text)))
    ).collect: _*
  )




  // Get total number n-grams used.
  val numTokens : Int = idf.size


  // Create n-gram to global index hashmap:
  //    Map(n-gram -> global index)
  private val globalIndex : HashMap[String, Int] = HashMap(
    idf.keys.zipWithIndex.toSeq
    : _*)

  // 5. Document Transformer: document => sparse tf-idf vector.
  // This takes a single document, tokenizes it, hashes it,
  // and finally returns a sparse vector containing the
  // tf-idf entries of the document n-grams (0 for all n-grams
  // not contained in the document).

  def transform(doc: String): Vector = {
    // Map(n-gram -> document tf)
    val hashedDoc = hash(tokenize(doc)).filter(e => idf.contains(e._1))
    Vectors.sparse(
      numTokens,
      hashedDoc.map {
        case (ngram, tf) => (globalIndex(ngram), idf(ngram) * tf)
      }.toArray
    )
  }


  // 6. Data Transformer: RDD[documents] => RDD[LabeledPoints]

  val transformedData: RDD[(LabeledPoint)] = {
    td.data.map(e => LabeledPoint(e.label, transform(e.text)))
  }


  // 7. Finally extract category map, associating label to category.
  val categoryMap = td.data.map(e => (e.label, e.category)).collectAsMap


  // 8. Finally consider the case where new document has no matching-ngrams.
  val majorityCategory = categoryMap.getOrElse(
    td.data.map(e => e.label).countByValue.maxBy(_._2)._1,
    ""
  )

}




