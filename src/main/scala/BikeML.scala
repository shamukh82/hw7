import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.evaluation.RegressionMetrics
import breeze.linalg._
import breeze.plot._

import vegas._
import vegas.render.WindowRenderer._
import vegas.sparkExt._
import org.apache.spark.sql.SparkSession;

/**
  * Created by smukherjee5 on 10/20/17.
  */
object BikeML {

  def main(args: Array[String]): Unit = {



    val spark = SparkSession
      .builder()
      .appName("AutoML")
        .master("local")
      .getOrCreate()



    //P1
    //Bring in the auto dataset into spark
    val hour = spark.read.option("header","true").csv("/Users/smukherjee5/harvard_hw/hw7/hour.csv")
    //nullValue='NA'
    val hourRecords = hour.select("season","yr","mnth","hr","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed","cnt")

    val categoriesSeason = hourRecords.rdd.map(r => r(0)).distinct.collect.zipWithIndex.toMap
    val categoriesYr = hourRecords.rdd.map(r => r(1)).distinct.collect.zipWithIndex.toMap
    val categoriesMon = hourRecords.rdd.map(r => r(2)).distinct.collect.zipWithIndex.toMap
    val categoriesHr = hourRecords.rdd.map(r => r(3)).distinct.collect.zipWithIndex.toMap
    val categoriesWeekDay = hourRecords.rdd.map(r => r(5)).distinct.collect.zipWithIndex.toMap
    val categoriesWorkDay = hourRecords.rdd.map(r => r(6)).distinct.collect.zipWithIndex.toMap



  }
}
