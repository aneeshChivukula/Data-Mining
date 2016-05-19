import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD._
import org.apache.spark.rdd._
import org.apache.spark.graphx._
import org.apache.spark.graphx.Edge
import org.apache.spark.graphx.VertexRDD
import java.io._
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.MutableList
import org.apache.spark.ml.feature.StringIndexer


object LoadPropertyGraph {
  def main(args: Array[String]) {
    
    val conf = new SparkConf().setAppName("Data Loader")
    val sc = new SparkContext(conf)
    
//		val sqlContext = new org.apache.spark.sql.SQLContext(sc)		
//		import sqlContext._
//		import sqlContext.implicits._
//		val myRDD = sc.parallelize(Array(
//		"one","two","three"
//		))
//		val df2 = myRDD.toDF
//		df2.show()
    
    
    
    val cspaperauthoraffiliations : RDD[(String, String)] =  sc.textFile("hdfs://atlas8:9000/data/cspaperauthoraffiliationspidaid.csv").map( line => line.split(",")).map(row => (row(0).toString, row(1).toString))
    val cspaperkeywords =  sc.textFile("hdfs://atlas8:9000/data/cspaperkeywordspidfidkn.csv")

    val cspapers : RDD[(String,String)] =  sc.textFile("hdfs://atlas8:9000/data/cspaperspidcsidmvn.csv").map( line => line.split(",")).map(row => (row(0).toString, row(1).toString))
    
    val conferenceinstances: RDD[(String,String)] =  sc.textFile("hdfs://atlas8:9000/data/conferenceinstancescsidcsd.csv").map( line => line.split(",")).map(row => (row(0).toString, row(1).toString))
    
    
    val cspapersauthorsconferencesjoin = cspaperauthoraffiliations.join(cspapers)
    cspapersauthorsconferencesjoin.cache()
    
    val cspapersauthorsconferencesdatesjoin = cspapersauthorsconferencesjoin.map( k =>
        (k._2._2,(k._2._1,k._1))
    ).join(conferenceinstances)
    cspapersauthorsconferencesdatesjoin.cache()
    
    val cspapersauthorsconferencesdatesjoin2015 = cspapersauthorsconferencesdatesjoin.filter(
        t=>
          t._2._2 contains "2015"
      )
    
    val vertexRDDWithIndex = cspapersauthorsconferencesdatesjoin2015.map(t => t._2._1._1).distinct().zipWithIndex
    vertexRDDWithIndex.cache()
    
    var vertexRDD = vertexRDDWithIndex.map{case (k,v) => (v,(k))}
    
    val authorspapers2015 = cspapersauthorsconferencesdatesjoin2015.map(t => (t._2._1._1, t._1)).groupByKey().join(vertexRDDWithIndex).sortBy(_._2._1.toSet.size,false).map(t => (t._2._2,(t._1,t._2._1.toSet)))
    val papersauthors2015 = cspapersauthorsconferencesdatesjoin2015.map(t => (t._1, t._2._1._1)).groupByKey()
    
    authorspapers2015.cache()
    papersauthors2015.cache()
    // PairRDDFunctions.aggregateByKey is high performant alternative to groupByKey()
    // PairRDDFunctions.reduceByKey is high performant alternative to groupByKey().mapValues()
//    println(papersauthors2015.count()) // 656
//    println(authorspapers2015.count()) // 251257
    
    
    val l:MutableList[RDD[(Long,Long,(String))]] = MutableList()
    
    val a = authorspapers2015.collect()
    for(i <- 0 until a.length){
      val t1 = a(i)
//    authorspapers2015.foreachPartition( partition => {
//      val a = partition.toArray
//      for(i <- 0 until a.length){
//      val t1 = a(i)
      val searchpapers = sc.parallelize(t1._2._2.toSeq).map(v=>(v,t1._1))
      
      var papersauthors2015idsnames = papersauthors2015.join(searchpapers)
      val b = papersauthors2015idsnames.collect()
      
      for(j <- 0 until b.length){
        val t2 = b(j)
        var currpapercoauthors = sc.parallelize(t2._2._1.toSeq).map(v=>(v,t2._2._2)).join(vertexRDDWithIndex).map(t => (t._2._1,t._2._2,(t._1)))
        l += currpapercoauthors
//      }
//      }
    }
//   )
    }
   // Reduce input data size. Rather than Index, Hash stringnames on vertices to get unique longs as vertex ids in graphx. Get tuples from data api. 
   // Data storage options that include graph queries include nosql datawarehouse, graph database, object database.
      // Could also look into data modelling, data storage, data visualization scenarios/issues in domain databases like xml databases, geospatial databases, traffic databases, 3d gaming databases
      // Must look into system-level issues like data benchmarks, data indexing, data partitioning and data storage for big data.
    
   val coauthors2015 = sc.union(l)
   coauthors2015.collect.take(10).foreach(println)   
      
    def makeedgelistclasses(e: (Long,Long,(String))):Edge[(String)] ={
       return new Edge(e._1,e._2,(e._3))
    }
    
    val edgeRDD: RDD[Edge[(String)]] =  coauthors2015.map(makeedgelistclasses)
    
    type MyVertex = (String)
    type MyEdge = (String)    
    val CoauthorshipNetwork2015: Graph[MyVertex, MyEdge] = Graph.apply(vertexRDD,edgeRDD)
      // All vertices must be uniquely identified with 64bit longs
    
    CoauthorshipNetwork2015.vertices.collect.foreach(println)
    CoauthorshipNetwork2015.edges.collect.foreach(println)
//     GraphOps Examples
//     http://ampcamp.berkeley.edu/big-data-mini-course/graph-analytics-with-graphx.html
//     http://spark.apache.org/docs/latest/graphx-programming-guide.html#vertex_and_edge_rdds
  }
 
}
/*
Spark-submit configuration :
spark-submit --class "LoadPropertyGraph" --master yarn --deploy-mode cluster --driver-memory 16G --executor-memory 16G --num-executors 14 --executor-cores 7 ~/data-loaders_2.11-1.0.jar
spark-submit --class "LoadPropertyGraph" --master yarn --deploy-mode cluster --driver-memory 25G --executor-memory 25G --num-executors 30 --executor-cores 7 ~/data-loaders_2.11-1.0.jar

ssh achivuku@atlas8
scp ./target/scala-2.11/data-loaders_2.11-1.0.jar achivuku@atlas8:~


HDFS Shell Commands :

hadoop fs -copyFromLocal cspaperauthoraffiliationspidaid.csv /data/
hadoop fs -copyFromLocal cspaperspidcsidmvn.csv /data/
hadoop fs -copyFromLocal cspaperkeywordspidfidkn.csv /data/
hadoop fs -copyFromLocal conferenceinstancescsidcsd.csv /data/

grep -Ev $',$' cspaperspidcsidmvn.csv > cspaperspidcsidmvn2.csv
grep -Ev $'\s' cspaperspidcsidmvn2.csv > cspaperspidcsidmvn3.csv
grep -Ev $'-' cspaperspidcsidmvn3.csv > cspaperspidcsidmvn4.csv
grep -Ev $'[a-z]' cspaperspidcsidmvn4.csv > cspaperspidcsidmvn5.csv
cp cspaperspidcsidmvn5.csv cspaperspidcsidmvn.csv
hadoop fs -put cspaperspidcsidmvn.csv /data/

sed 's/\[//g' conferenceinstancescsidcsd.csv > conferenceinstancescsidcsd2.csv
grep -Ev $'^,' conferenceinstancescsidcsd2.csv > conferenceinstancescsidcsd3.csv
sed 's/\]//g' conferenceinstancescsidcsd3.csv > conferenceinstancescsidcsd4.csv
grep -Ev $',$' conferenceinstancescsidcsd4.csv > conferenceinstancescsidcsd5.csv
cp conferenceinstancescsidcsd5.csv conferenceinstancescsidcsd.csv 
hadoop fs -rmr /data/conferenceinstancescsidcsd.csv
hadoop fs -copyFromLocal conferenceinstancescsidcsd.csv /data/conferenceinstancescsidcsd.csv
     
Compared to SQL joins, RDD joins are significantly faster 
Join operation is defined only on PairwiseRDDs which are quite different from a relation / table in SQL. Each element of PairwiseRDD is a Tuple2 where the first element is the key and the second is value. Both can contain complex objects as long as key provides a meaningful hashCode
http://stackoverflow.com/questions/33321704/join-two-rdd-in-spark

SQL Context API
https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.sql.SQLContext
RDD API
http://spark.apache.org/docs/1.1.1/api/scala/index.html#org.apache.spark.rdd.RDD
SchemaRDD (Spark 1.1.1 JavaDoc)
https://spark.apache.org/docs/1.1.0/api/java/org/apache/spark/sql/SchemaRDD.html
Spark PairedRDD API
https://spark.apache.org/docs/0.6.2/api/core/spark/PairRDDFunctions.html
https://spark.apache.org/docs/1.6.0/api/java/org/apache/spark/rdd/PairRDDFunctions.html
https://www.safaribooksonline.com/library/view/learning-spark/9781449359034/ch04.html
Scala Collections API
http://www.scala-lang.org/docu/files/collections-api/collections.html
http://www.scala-lang.org/api/2.7.6/scala/Iterator.html
GraphX API
http://spark.apache.org/docs/1.1.1/api/scala/index.html#org.apache.spark.graphx.Graph$
Spark DataFrame API
http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.sql.DataFrame
GraphOps Examples
http://ampcamp.berkeley.edu/big-data-mini-course/graph-analytics-with-graphx.html
GraphFrames API Docs
http://graphframes.github.io/api/scala/index.html#org.graphframes.GraphFrame$
Graph Analytics in Spark
http://www.slideshare.net/pacoid/graph-analytics-in-spark
Top 5 features released in spark 1.6
http://kirillpavlov.com/blog/2016/02/21/top-5-features-released-in-spark-1.6/
Scala Code Review: foldLeft and foldRight
https://oldfashionedsoftware.com/2009/07/10/scala-code-review-foldleft-and-foldright/


Searching for date by string and regex matching. Use java.sql.Date with groupBy on pairedRDDs for fine grained date manipulation.
(start/86400 to end/86400).map(day => (day, rdd.filter(rec => rec.time 
>= day*86400 && rec.time < (day+1)*86400))) 

groupByKey must be able to hold all the key-value pairs for any key in memory. If a key has too many values, it can result in an OutOfMemoryError



*/


/*
Temporary Code Snippets :

    println(edgeRDDArray(0))
    edgeRDDArray.foreach(
        a =>
        println(a._1)
        )


//    edgeRDDArray.foreach(t =>
//      t.foreach(
//          a=>
//            println(a)
//          )
//    )
    
    
    
//    edgeRDDArray.foreach( t =>
//      println(t)
//    )
    
//    edgeRDDArray.foreach(
//      t=>
//        t.foreach(
//            a=>
//      println(a)
//      for(j <- 0 until a.length){
//        println(a(j))
//      }    
//    )
//    )
//    val file = new File("/home/achivuku/output/filteredcspublications/rddoutputs/cspapersauthorsconferencesdatesjoin2015.csv")
    
//    for(i <- 0 until edgeRDDArray.length){
//    }
    
    
    // http://stackoverflow.com/questions/24497389/using-scala-to-dump-result-processed-by-spark-to-hdfs
    // http://stackoverflow.com/questions/4604237/how-to-write-to-a-file-in-scala    
    
    
//    edgeRDDArray.saveAsTextFile("file:///home/achivuku/output/filteredcspublications/rddinputs/cspapersauthorsconferencesdatesjoin2015")
    
//    val edgeRDD: RDD[(String,String,String)] =  sc.textFile("file:///home/achivuku/output/filteredcspublications/rddinputs/cspapersauthorsconferencesdatesjoin2015/").map( line => line.split(",")).map(row => (row(0).toString, row(1).toString, row(2).toString))
//    val CoauthorshipNetwork2015: Graph[String, (String,String,(String))] = Graph.apply(vertexRDD,edgeRDD, "DUMMY")
    // Array cannot be there in RHS. Only an RDD of strings
    
//    for((x,i) <- xs.view.zipWithIndex) println("String #" + i + " is " + x)

    
//    val edgeRDD =  cspapersauthorsconferencesdatesjoin2015.map(t => t._2._1._1).toArray()  
        
    
//    cspapersauthorsconferencesjoin.map(k => k._1).collect().take(1).foreach(println)
//    cspapersauthorsconferencesjoin.map(k => k._2._1).collect().take(1).foreach(println)
//    cspapersauthorsconferencesjoin.map(k => k._2._2).collect().take(1).foreach(println)
//    cspapersauthorsconferencesjoin.collect().take(1).foreach(println)
    
//    val conferenceinstancesnomissing = conferenceinstances.filter(line => !(line.split(',')(0).replaceAll("\\[","") == ""))
//    conferenceinstancesnomissing.cache()
//    println("conferenceinstances.count() " + conferenceinstances.count())
//    println("conferenceinstancesnomissing.count() " + conferenceinstancesnomissing.count())

//    val cspapersauthorsconferencesreduce = (cspaperauthoraffiliations union cspapers).reduceByKey(_ ++ _) 
//    println("cspapers.count() " + cspapers.count())
//    println("cspaperauthoraffiliations.count() " + cspaperauthoraffiliations.count())
//    println("cspapersauthorsconferencesreduce.count() " + cspapersauthorsconferencesreduce.count())
//    println("cspapers.first() " + cspapers.first())
//    cspapersauthorsconferencesjoin.collect().foreach(println)
//    cspapersauthorsconferences.map(t => sline = (t(2), t(0), t(1)))
    
//    def removemissingvalues(line: String):Boolean ={
//      return !(line.split(',')(0).replaceAll("\\[","") == "")
//    }
//    val rdd1 = conferenceinstances.filter(removemissingvalues)
//    rdd1.collect().foreach(println)

//    def makeedgelistarray(t: (String,Array[Iterable[String]])):Array[Edge[(String,String,(String))]] ={
//          val a:Array[Edge[(String,String,(String))]] = new Array[Edge[(String,String,(String))]](t._2.length * t._2.length)
//          for(i <- 0 until t._2.length){
//              for(j <- i+1 until t._2.length){
//                a:+ (t._2(i),t._2(j),(t._1))
//                
//              }
//            }
//          return a
//    }  
    
    
//    val rdd1 = conferenceinstances.flatMap(x => 
//      if(x.split(",")(0) != null)
//        x
//      else
//        None    
//    )
//    rdd1.collect().foreach(println)
//    val rdd1 = conferenceinstances.map(new Helpers().removemissingvalues())


//    edgeListJoin.take(1).foreach(
//    compactbuffer => 
//      for((a,i) <- compactbuffer._2.iterator.zipWithIndex){
//        println(a,i)
//        for((b,j) <- compactbuffer._2.iterator.zipWithIndex){
//          if(j>i){
//            println((a,b,compactbuffer._1))
//          }
//        }
//      }
////      compactbuffer._2.iterator.hasNext      
//    )






/*
    val cspapersauthorsconferencesdatesjoin2014 = cspapersauthorsconferencesdatesjoin.filter(
        t=>
          t._2._2 contains "2014"
      )
      
    val cspapersauthorsconferencesdatesjoin2013 = cspapersauthorsconferencesdatesjoin.filter(
        t=>
          t._2._2 contains "2013"
      )

    val cspapersauthorsconferencesdatesjoin2012 = cspapersauthorsconferencesdatesjoin.filter(
        t=>
          t._2._2 contains "2012"
      )

    val cspapersauthorsconferencesdatesjoin2011 = cspapersauthorsconferencesdatesjoin.filter(
        t=>
          t._2._2 contains "2011"
      )

      
    val cspapersauthorsconferencesdatesjoin2010 = cspapersauthorsconferencesdatesjoin.filter(
        t=>
          t._2._2 contains "2010"
      )

    val cspapersauthorsconferencesdatesjoin2009 = cspapersauthorsconferencesdatesjoin.filter(
        t=>
          t._2._2 contains "2009"
      )

    val cspapersauthorsconferencesdatesjoin2008 = cspapersauthorsconferencesdatesjoin.filter(
        t=>
          t._2._2 contains "2008"
      )
      
    val cspapersauthorsconferencesdatesjoin2007 = cspapersauthorsconferencesdatesjoin.filter(
        t=>
          t._2._2 contains "2007"
      )
      
    val cspapersauthorsconferencesdatesjoin2006 = cspapersauthorsconferencesdatesjoin.filter(
        t=>
          t._2._2 contains "2006"
      )

    val cspapersauthorsconferencesdatesjoin2005 = cspapersauthorsconferencesdatesjoin.filter(
        t=>
          t._2._2 contains "2005"
      )

    val cspapersauthorsconferencesdatesjoin2004 = cspapersauthorsconferencesdatesjoin.filter(
        t=>
          t._2._2 contains "2004"
      )
 */     
//    cspapersauthorsconferencesdatesjoin2015.collect.foreach(println)














//    cspapersauthorsconferencesdatesjoin2015.take(1).foreach(println)    
//    val df = cspapersauthorsconferencesdatesjoin2015.toDF("row", "col")
    
    // Continue from here
      // After partitioning on date, take only papersids authorsids in rdd. And do two groupby followed by intersect to get paperid
      // Must loop over all combinations of authors starting from most cited paper
      // Must not repeat intersections over authors that have already been scanned by looping over authors than papers
      // To be able to load into graphx, loop by index than id
      
    // cspapersauthorsconferencesdatesjoin2015.collect.take(1).foreach(println)
    // (44BC5EE8,((82B4E676,7F53C29E),2015/05/18))

















    // (7BE5F0C8,(CompactBuffer(4708B833, 447B8E60, 461B085B, 439F03CF, 42E6BE19, 42E6BE19, 42E6BE19, 42E6BE19, 42E6BE19, 42E6BE19, 44E5A1B2, 44D3E57F, 44D3E57F, 44D3E57F),0))
    // (7FF56C9D,(CompactBuffer(42C509C4, 42C509C4, 42C509C4, 42C509C4, 42C509C4, 42C509C4, 42C509C4, 42C509C4, 42C509C4, 42C509C4, 42C509C4, 42C509C4, 42C509C4, 42C509C4, 42C509C4, 42C509C4, 42C509C4, 42C509C4, 42C509C4, 46C53898, 46C53898, 46C53898, 46C53898, 469D149A, 469D149A, 469D149A, 469D149A, 469D149A, 469D149A, 469D149A, 469D149A, 469D149A, 469D149A, 442BD7CD, 45610CDA, 45610CDA, 45610CDA, 45610CDA, 45610CDA, 45610CDA, 45610CDA, 455B6732, 455B6732, 455B6732, 455B6732, 455B6732, 455B6732, 455B6732, 455B6732, 47B7ABEF, 47B7ABEF, 47B7ABEF, 47B7ABEF, 4737FD14, 436150FA, 43AA5802, 44233F91, 44233F91, 44233F91, 44233F91, 44233F91, 44233F91, 44233F91, 44233F91, 44233F91, 44233F91, 44233F91, 44233F91, 44233F91, 4569270C, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 4306EF42, 47169EC5, 47169EC5, 47169EC5, 47169EC5, 47169EC5, 457D0954, 457D0954, 457D0954, 457D0954, 457D0954, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 46FBD930, 4789F291, 4789F291, 4789F291, 4789F291, 4789F291, 4789F291, 4789F291, 4789F291, 4789F291, 4789F291, 4789F291, 4789F291, 4789F291, 4789F291, 4789F291, 4789F291, 4789F291, 4789F291, 4789F291, 4523D0A5, 4523D0A5, 4523D0A5, 4523D0A5, 4523D0A5, 4523D0A5, 4523D0A5, 4523D0A5, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 44AB61E1, 453F3F1D, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 47B51FBD, 4390334E, 460452E7, 460452E7, 460452E7, 460452E7, 460452E7, 460452E7, 460452E7, 460452E7, 460452E7, 460452E7, 460452E7, 460452E7, 460452E7, 460452E7, 45827814, 470CB4A0, 4708B833, 4708B833),96391))    
//    val authorspapers2015indexescounts = authorspapers2015.join(authorsindexes2015)



















//    authorspapers2015.collect.take(1).foreach(t=>
//      println(t._1))
//    authorspapers2015.collect.take(1).foreach(t=>
//      println(t._2))
    
//    def makeedgelisttuples3(t: (Long,Iterable[String])):Array[(Long,Long,String)] ={
//        println("In makeedgelisttuples2")
//
//        var currpapercoauthors = sc.parallelize(t._2.toSeq).map(v=>(v,t._1)).join(vertexRDDWithIndex).map(t => (t._2._1,t._2._2,(t._1))).collect()
//        return currpapercoauthors
//    }
//    
//    
//    def makeedgelisttuples2(t: (String,Set[String])):Array[(Long,Long,(String))] ={
//      
//      println("In makeedgelisttuples2")
//      
//      val searchpapers = sc.parallelize(t._2.toSeq).map(v=>(v,t._1))
//      var allpapercoauthors = papersauthors2015.join(searchpapers).flatMap(t=>makeedgelisttuples3((t._2._2.toLong,t._2._1))).distinct.collect()
//      
//      searchpapers.collect.foreach(println)
//      
//      return allpapercoauthors
//    }

//    val coauthors2015  










//    val coauthors2015 = authorspapers2015.flatMapValues(makeedgelisttuples2).map(t=> t._2)
    // Return coauthors for current author
//    coauthors2015.collect.take(1).foreach(println)









    val vertexRDDWithIndex = cspapersauthorsconferencesdatesjoin2015.map(t => t._2._1._1).distinct().zipWithIndex
    vertexRDDWithIndex.cache()
    var vertexRDDarray =  vertexRDDWithIndex.map{case (k,v) => (v,(k))}.collect()
    var vertexRDD = sc.parallelize(vertexRDDarray)
//    vertexRDD.collect.foreach(println)
    
    
    
    
    
    
    def makeedgelisttuples(t: (String,Iterable[String])):MutableList[(Long,Long,(String))] ={
          val it = t._2.iterator
          val l:MutableList[(Long,Long,(String))] = MutableList()
          for((a,i) <- it.zipWithIndex){
              val avidx = vertexRDDWithIndex.lookup(a)(0)
              println("avidx "+avidx)
              for((b,j) <- it.zipWithIndex){
                if(j==i+1){
//                if(j>i){
                  val bvidx = vertexRDDWithIndex.lookup(b)(0)
                  println("bvidx "+bvidx)
                  l += Tuple3(avidx,bvidx,(t._1))
                }
              }
            }
//          print("l.length"+l.length)
          return l
    }  




















//    // There is no alternative to using a Array for packing and unpacking tuples incrementally after a join
//    // Is groupby and lookup the best way to generate edge lists?
//      // Drop multigraph creation. Create adjacency matrix only for zero elements.
//      // Think about key values suitable for combining keys.
//      // Work with small size of features and sample before distributed deployment in spark rdds.
//    


















    var edgeListJoin = cspapersauthorsconferencesdatesjoin2015.map(t => (t._1, t._2._1._1)).groupByKey().collect()
    var edgeRDDArray = edgeListJoin.flatMap(makeedgelisttuples).map(makeedgelistclasses)

















Try set intersection with groupby to get papers common between vertices.Otherwise Write code using mllib data types and convert table to adjacency list outside RDD. 
Check output is seen for creation of property graph.
check ways to increase performance in spark submit
(
groupby
mapValues, countByKey, sortBy for sorting
combineByKey or flatMapValues to loop over auth id
cogroup or join for unique value list lookup
reduceByKey or collectAsMap or lookup to get output
foldByKey to get values in max-min ranges or take a union all
RDD ops

Experimental:
combineByKeyWithClassTag
aggregateByKey

Note that this method should only be used if the resulting map is expected to be small, as the whole thing is loaded into the driver's memory. To handle very large results, consider using rdd.mapValues(_ => 1L).reduceByKey(_ + _), which returns an RDD[T, Long] instead of a map.

Note: As currently implemented, groupByKey must be able to hold all the key-value pairs for any key in memory. If a key has too many values, it can result in an OutOfMemoryError.

Note: This operation may be very expensive. If you are grouping in order to perform an aggregation (such as a sum or average) over each key, using PairRDDFunctions.aggregateByKey or PairRDDFunctions.reduceByKey will provide much better performance.

If you find yourself writing code where you groupByKey() and then use a reduce() or fold() on the values, you can probably achieve the same result more efficiently by using one of the per-key aggregation functions. Rather than reducing the RDD to an in-memory value, we reduce the data per key and get back an RDD with the reduced values corresponding to each key. For example, rdd.reduceByKey(func) produces the same RDD as rdd.groupByKey().mapValues(value => value.reduce(func)) but is more efficient as it avoids the step of creating a list of values for each key.

flatMapValues giving org.apache.spark.SparkException: Task not serializable
https://databricks.gitbooks.io/databricks-spark-knowledge-base/content/troubleshooting/javaionotserializableexception.html
http://stackoverflow.com/questions/22592811/task-not-serializable-java-io-notserializableexception-when-calling-function-ou
https://github.com/jaceklaskowski/mastering-apache-spark-book/blob/master/spark-tips-and-tricks-sparkexception-task-not-serializable.adoc
http://stackoverflow.com/questions/29295838/org-apache-spark-sparkexception-task-not-serializable

Dont know what to do. Using loops instead of methods and reducing input size.
)


// There is no alternative to using a Array for packing and unpacking tuples incrementally after a join
// Is groupby and lookup the best way to generate edge lists? Each Lookup taking around half a minute. Around 2.5 lakhs lookups were done in  2.2 h before job was killed 
// Drop multigraph creation. Create adjacency matrix only for zero elements.
// Check performance with small size of features and sample before distributed deployment in spark rdds.
Complete this work in few days by fri

its more of a bipartitite to adjacency matrix problem where one dimension is projected or folded away
looping on iterators cannot be affected. but lookups can be reduced by two groupbykey followed by set intersection of values.
improve lookup, indexing, sorting key-values
other alternative is to convert table to adjacency matrix inside spark and adjacency matrix to edge list outside spark
final alternative is to reduce input data size

authidx, authid - paired rdd lookup
paperid : author ids - groupby
inside spark think about key values suitable for combining keys over set intersections - lookup&intersect paperids for every authorid belonging to paperid - reduce input data size for allowing parallel processing

After partitioning on date, take only papersids authorsids in rdd. And do two groupby followed by intersect to get paperid
Must loop over all combinations of authors starting from most cited paper
Must not repeat intersections over authors that have already been scanned by looping over authors than papers
To be able to load into graphx, loop by index than id

adjacency matrix can be created in spark mllib data types, data matrices, data indexing. then convert adjacency matrix to edge list outside spark. - spark mllib data types have dependency on sqlcontext but not hive - sparksql dependency works only for dataframes created from existing rdd that do not access hive tables in database
graphframes is alternative to mllib data types

otherwise do all table to matrix to list conversion outside spark rdds if size permits






*/