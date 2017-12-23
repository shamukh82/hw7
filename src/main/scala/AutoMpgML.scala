//package com.hw7.ml
//
//
//import org.apache.spark.SparkContext
//import org.apache.spark.SparkConf
//import org.apache.spark.mllib.regression.LabeledPoint
//import org.apache.spark.mllib.linalg.Vectors
//import org.apache.spark.mllib.regression.LinearRegressionWithSGD
//import org.apache.spark.mllib.evaluation.RegressionMetrics
//import breeze.linalg._
//import breeze.plot._
//
//import vegas._
//import vegas.render.WindowRenderer._
//import vegas.sparkExt._
//
///**
//  * Created by smukherjee5 on 10/20/17.
//  */
//object AutoMpgML {
//
//  def main(args: Array[String]): Unit = {
//
//
//
////    val spark = SparkSession
////      .builder()
////      .appName("AutoML")
////        .master("local")
////      .getOrCreate()
////
////    val auto = spark.read.csv("/Users/smukherjee5/harvard_hw/hw7/auto_mpg_original-1.csv",sep)
//
//
//    val conf = new SparkConf().setMaster("local[*]").setAppName("AutoMpgML")
//    val sparkContext = new SparkContext(conf)
//    val sqlContext= new org.apache.spark.sql.SQLContext(sparkContext)
//    import sqlContext.implicits._
//
//    //P1
//    //Bring in the auto dataset into spark
//    val auto = sparkContext.textFile("/Users/smukherjee5/harvard_hw/hw7/auto_mpg_original-1.csv")
//    //nullValue='NA'
//
//    //Named columns using case class defined
//    val autoDF = auto.map(_.split(",")).map(p=> Auto(p(0).toString,p(1).toString,p(2).toString,p(3).toString,p(4).toString,p(5).toString,p(6).toString,p(7).toString,p(8).toString))
//
//    //Now do a 80:20 split of data for training:test
//    val Array(trainData,testData) = autoDF.randomSplit(Array(0.8,0.2), seed=11L)
//
//    //Create views to query from using spark sql from DF
//    trainData.toDF.createOrReplaceTempView("trainData")
//    testData.toDF.createOrReplaceTempView("testData")
//
//    //println("autoDF count: "+autoDF.count())
//
//    //Get avg values for HP and mpg in the train data split
//    val avgTrainDataHP = sqlContext.sql("select mean(horsepower) from trainData").rdd.coalesce(1).take(1).mkString(",").replace("[","").replace("]","").toDouble
//    val avgTrainDataMpg = sqlContext.sql("select mean(mpg) from trainData").rdd.coalesce(1).take(1).mkString(",").replace("[","").replace("]","").toDouble
//
//    //Get avg values for HP and mpg in the test data split
//    val avgTestDataHP = sqlContext.sql("select mean(horsepower) from testData").rdd.coalesce(1).take(1).mkString(",").replace("[","").replace("]","").toDouble
//    val avgTestDataMpg = sqlContext.sql("select mean(mpg) from testData").rdd.coalesce(1).take(1).mkString(",").replace("[","").replace("]","").toDouble
//
//    //Replace null values which I see as NA
//    val nullRemovedTrainDataSet = sqlContext.sql("select regexp_replace(horsepower,'NA','"+avgTrainDataHP+"') horsepower,regexp_replace(mpg,'NA','"+avgTrainDataMpg+"') mpg,cylinders,displacement,weight,acceleration,modelYear,origin,carName from trainData")
//    val nullRemovedTestDataSet = sqlContext.sql("select regexp_replace(horsepower,'NA','"+avgTestDataHP+"') horsepower,regexp_replace(mpg,'NA','"+avgTestDataMpg+"') mpg,cylinders,displacement,weight,acceleration,modelYear,origin,carName from testData")
//
//    //P2
//    //Treat mpg as a feature and horsepower as the target variable (label)
//
////    scala> nullRemovedTrainDataSet.show(10)
////      +----------+---+---------+------------+------+------------+---------+------+---------+
////      |horsepower|mpg|cylinders|displacement|weight|acceleration|modelYear|origin|  carName|
////      +----------+---+---------+------------+------+------------+---------+------+---------+
////      |       130| 18|        8|         307|  3504|          12|       70|     1|chevrolet|
////      |       165| 15|        8|         350|  3693|        11.5|       70|     1|    buick|
////      |       150| 18|        8|         318|  3436|          11|       70|     1| plymouth|
////      |       140| 17|        8|         302|  3449|        10.5|       70|     1|     ford|
////      |       198| 15|        8|         429|  4341|          10|       70|     1|     ford|
////      |       220| 14|        8|         454|  4354|           9|       70|     1|chevrolet|
////      |       215| 14|        8|         440|  4312|         8.5|       70|     1| plymouth|
////      |       225| 14|        8|         455|  4425|          10|       70|     1|  pontiac|
////      |       153| 27|        8|         351|  4034|          11|       70|     1|     ford|
////      |       175| 28|        8|         383|  4166|        10.5|       70|     1| plymouth|
////      +----------+---+---------+------------+------+------------+---------+------+---------+
////
//    //val data = nullRemovedTrainDataSet.rdd.map(r => LabeledPoint(java.lang.Double.parseDouble(r(0).toString) ,Vectors.dense(java.lang.Double.parseDouble(r(1).toString))))
//    val data = nullRemovedTrainDataSet.rdd.map(r => LabeledPoint((r(0).toString).toDouble ,Vectors.dense(r(1).toString.toDouble)))
//
////    scala> data.collect
////    res166: Array[org.apache.spark.mllib.regression.LabeledPoint] = Array((130.0,[18.0]), (165.0,[15.0]), (150.0,[18.0]), (140.0,[17.0]), (198.0,[15.0]), (220.0,[14.0]), (215.0,[14.0]), (225.0,[14.0]), (153.0,[27.0]), (175.0,[28.0]), (175.0,[25.0]), (170.0,[15.0]), (160.0,[14.0]), (140.0,[16.0]), (150.0,[15.0]), (225.0,[14.0]), (95.0,[24.0]), (97.0,[18.0]), (88.0,[27.0]), (46.0,[26.0]), (87.0,[25.0]), (90.0,[24.0]), (95.0,[25.0]), (113.0,[26.0]), (90.0,[21.0]), (215.0,[10.0]), (200.0,[10.0]), (210.0,[11.0]), (193.0,[9.0]), (88.0,[27.0]), (90.0,[28.0]), (95.0,[25.0]), (87.0,[25.0]), (100.0,[19.0]), (88.0,[19.0]), (100.0,[18.0]), (165.0,[14.0]), (175.0,[14.0]), (150.0,[14.0]), (180.0,[12.0]), (170.0,[13.0]), (175.0,[13.0]), (110.0,[18.0]), (72.0,[22.0]), (100.0,[19.0]), (88.0,[18.0]), (86.0,[...
////    scala> val data = nullRemovedTrainDataSet.rdd.map(r => LabeledPoint(java.lang.Double.parseDouble(r(0).toString) ,Vectors.dense(java.lang.Double.parseDouble(r(1).toString))))
////    data: org.apache.spark.rdd.RDD[org.apache.spark.mllib.regression.LabeledPoint] = MapPartitionsRDD[482] at map at <console>:32
////
//
//    data.cache
//
//    val stepSize = 0.1
//    val numIterations = 150
//    val model = LinearRegressionWithSGD.train(data,numIterations, stepSize)
//
//
//    val testDataPoints = nullRemovedTestDataSet.rdd.map(r => LabeledPoint((r(0).toString).toDouble ,Vectors.dense(r(1).toString.toDouble)))
//
//    val valuesAndPreds = testDataPoints.map { point =>
//      val prediction = model.predict(point.features)
//      (point.label, prediction)
//    }
//
//    //val MSE = valuesAndPreds.map{ case(v, p) => math.pow((v - p), 2) }.mean()
//
//    val metrics = new RegressionMetrics(valuesAndPreds)
//
//    println("training Mean Squared Error = " + metrics.meanSquaredError)
//    println("training Mean Squared Error = " + metrics.meanAbsoluteError)
//
//    //After playing with various combinations
//    //scala>     println("training Mean Squared Error = " + MSE)
//    //training Mean Squared Error = 1.2506605251613675E254
//
//
//
//    val f = Figure()
//    val plt = f.subplot(0)
//
//    //Couldnt get this to work completely
//    plt += plot(valuesAndPreds.toDF("horsepower","mpg") , valuesAndPreds.toDF("horsepower","mpg"))
//    plt.xlabel = "horsepower axis"
//    plt.ylabel = "mpg axis"
//    f.saveas("lines.png")
//
//  }
//}
//case class Auto(mpg:String,cylinders:String,displacement:String,horsepower:String,weight:String,acceleration:String,modelYear:String,origin:String,carName:String)
