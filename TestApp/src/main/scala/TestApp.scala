import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object TestApp {
	def main(args: Array[String]) {
		val conf = new SparkConf().setAppName("Data Loader")
		val sc = new SparkContext(conf)
		val sqlContext = new org.apache.spark.sql.SQLContext(sc)		
		import sqlContext._
		import sqlContext.implicits._

		val myRDD = sc.parallelize(Array(
		"one","two","three"
		))
		
		val df = sqlContext.read.json("hdfs://atlas8:9000/data/markers.json")
		df.show()
		
		val df2 = myRDD.toDF
		df2.show()

	}

}
