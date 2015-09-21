package org.template.textclassification

import java.io.{InputStreamReader, BufferedReader, ByteArrayInputStream, Serializable}

import BIDMat.{CMat,CSMat,DMat,Dict,FMat,FND,GMat,GDMat,GIMat,GLMat,GSMat,GSDMat,HMat,IDict,Image,IMat,LMat,Mat,SMat,SBMat,SDMat}
import BIDMat.MatFunctions._
import BIDMat.SciFunctions._
import BIDMat.Solvers._
import BIDMat.Plotting._
import BIDMach.Learner
import BIDMach.models.{FM,GLM,KMeans,KMeansw,LDA,LDAgibbs,Model,NMF,SFA,RandomForest}
import BIDMach.networks.{DNN}
import BIDMach.datasources.{DataSource,MatDS,FilesDS,SFilesDS}
import BIDMach.mixins.{CosineSim,Perplexity,Top,L1Regularizer,L2Regularizer}
import BIDMach.updaters.{ADAGrad,Batch,BatchNorm,IncMult,IncNorm,Telescoping}
import BIDMach.causal.{IPTW}

import io.prediction.controller.{P2LAlgorithm, Params}
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

case class BIDMachLRAlgorithmParams (
                               regParam  : Double
                               ) extends Params


class BIDMachLRAlgorithm(
                           val sap: BIDMachLRAlgorithmParams
                           ) extends P2LAlgorithm[PreparedData, NativeLRModel, Query, PredictedResult] {
  // Train your model.
  def train(sc: SparkContext, pd: PreparedData): NativeLRModel = {
    new BIDMachLRModel(sc, pd, sap.regParam)
  }

  // Prediction method for trained model.
  def predict(model: NativeLRModel, query: Query): PredictedResult = {
    model.predict(query.text)
  }

}

  class BIDMachLRModel (
                  sc : SparkContext,
                  pd : PreparedData,
                  regParam : Double
                  ) extends Serializable with NativeLRModel {

    private val labels: Seq[Double] = pd.categoryMap.keys.toSeq

    val data = prepareDataFrame(sc, pd, labels)

    private val lrModels = fitLRModels

    def fitLRModels:Seq[(Double, LREstimate)] = {

      Mat.checkMKL
      Mat.checkCUDA
      if (Mat.hasCUDA > 0) GPUmem

      // 3. Create a logistic regression model for each class.
      val lrModels: Seq[(Double, LREstimate)] = labels.map(
        label => {
          val lab = label.toInt.toString

          val (categories, features) = getFMatsFromData(lab, data)

          val mm: Learner = trainGLM(features, FMat(categories))

          test(categories, features, mm)
          val modelmat = FMat(mm.modelmat)
          val weightSize = size(modelmat)._2 -1

          val weights = modelmat(1,0 to weightSize)

          val weightArray = (for(i <- 0 to weightSize -1) yield weights(0,i).toDouble).toArray

          // Return (label, feature coefficients, and intercept term.
          (label, LREstimate(weightArray, weights(0,weightSize)))
        }
      )
      lrModels
    }

    def predict(text : String): PredictedResult = {
      predict(text, pd, lrModels)
    }

    def trainGLM(traindata:SMat, traincats: FMat): Learner = {
      //min(traindata, 1, traindata) // the first "traindata" argument is the input, the other is output

      val (mm, mopts) = GLM.learner(traindata, traincats, GLM.logistic)
      mopts.what

      mopts.lrate = 0.1
      mopts.reg1weight = regParam
      mopts.batchSize = 1000
      mopts.npasses = 250
      mopts.autoReset = false
      mopts.addConstFeat = true
      mm.train
      mm
    }

    def getFMatsFromData(lab: String, data:DataFrame): (FMat, SMat) = {
      val features = data.select(lab, "features")

      val sparseVectorsWithRowIndices = (for (r <- features) yield (r.getAs[SparseVector](1), r.getAs[Double](0))).zipWithIndex 

      val triples = for {
        ((vector, innerLabel), rowIndex) <- sparseVectorsWithRowIndices
        (index, value) <- vector.indices zip vector.values
      }  yield ((rowIndex.toInt,index,value), innerLabel)

      val catTriples = for {
        ((vector, innerLabel), rowIndex) <- sparseVectorsWithRowIndices
      } yield (rowIndex.toInt,innerLabel.toInt,1.0)

      val cats = catTriples
      val feats = triples.map(x => x._1)

      val numRows = cats.count().toInt

      val catsMat = loadFMatTxt(cats,numRows)

      val featsMat = loadFMatTxt(feats,numRows)

      println(featsMat)

      (full(catsMat), featsMat)
    }

    //See https://github.com/BIDData/BIDMat/blob/master/src/main/scala/BIDMat/HMat.scala , method loadDMatTxt
    def loadFMatTxt(cats:RDD[(Int,Int,Double)], nrows: Int):SMat = {

      val rows = cats.map(x=> x._1).collect()
      val cols = cats.map(x=> x._2).collect()
      val vals = cats.map(x=> x._3).collect()


      println("LOADING")

      sparse(icol(cols.toList),icol(rows.toList),col(vals.toList))
    }

    def test(categories: DMat, features: SMat, mm: Learner): Unit = {
      val testdata = features
      val testcats = categories

      //min(testdata, 1, testdata)

      val predcats = zeros(testcats.nrows, testcats.ncols)



      val (nn, nopts) = GLM.predictor(mm.model, testdata, predcats)



      nopts.addConstFeat = true
      nn.predict


      computeAccuracy(FMat(testcats), predcats)
    }

    def computeAccuracy(testcats: FMat, predcats: FMat): Unit = {
      //println(testcats)
      //println(predcats)

      val lacc = (predcats ∙→ testcats + (1 - predcats) ∙→ (1 - testcats)) / predcats.ncols
      lacc.t
      println(mean(lacc))
    }

}
