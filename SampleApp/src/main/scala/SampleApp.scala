/* SampleApp.scala:
   This application simply counts the number of lines that contain "val" from itself
 */
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.graphx._

object SampleApp {
  def main(args: Array[String]) {
    val txtFile = "/home/aneesh/SampleApp/src/main/scala/SampleApp.scala"
    val conf = new SparkConf().setAppName("Sample Application").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val txtFileLines = sc.textFile(txtFile , 2).cache()
    val numAs = txtFileLines .filter(line => line.contains("val")).count()
    println("Lines with val: %s".format(numAs))
  }
}

/*
Use Eclipse to modify the project and test it
Use the sbt package to create the final jar
Deploy using spark-submit
Go to step 1, if necessary, and refine further

http://backtobazics.com/big-data/6-steps-to-setup-apache-spark-1-0-1-multi-node-cluster-on-centos/
http://backtobazics.com/big-data/spark/building-spark-application-jar-using-scala-and-sbt/

http://www.nodalpoint.com/development-and-deployment-of-spark-applications-with-scala-eclipse-and-sbt-part-1-installation-configuration/
http://www.nodalpoint.com/development-and-deployment-of-spark-applications-with-scala-eclipse-and-sbt-part-2-a-recommender-system/

spark-submit --class "SampleApp" --master local[2] target/scala-2.11/sample-project_2.11-1.0.jar

* */
