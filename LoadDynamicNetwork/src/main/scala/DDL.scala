import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf


object DDL {
	def main(args: Array[String]) {
		
	  
	  
		// Create database
		val conf = new SparkConf().setAppName("Data Loader")
		val sc = new SparkContext(conf)
		val sqlContext = new org.apache.spark.sql.SQLContext(sc)
		print(sqlContext)
		
		sqlContext.sql("show databases").collect().foreach(println)
    sqlContext.sql("drop database mydb cascade")
		sqlContext.sql("import sqlContext.implicits._")
		sqlContext.sql("show databases").collect().foreach(println)
		sqlContext.sql("CREATE DATABASE mydb")
		
		sqlContext.sql("USE mydb")
		sqlContext.sql("show tables").collect().foreach(println);

		
		
		
		
		
		// Create data frame of CS conference names and journal names
		sqlContext.sql("CREATE TABLE Papers (PaperID String,OriginalPaperTitle String,NormalizedPaperTitle String,PaperPublishYear String,PaperPublishDate String,PaperDocumentObjectIdentifier String,OriginalVenueName String,NormalizedVenueName String,JournalIDMappedToVenueName String,ConferenceSeriesIDMappedToVenueName String,PaperRank String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
		sqlContext.sql("LOAD DATA INPATH '/data/Papers.txt' OVERWRITE INTO TABLE Papers")
		var df = sqlContext.sql("select * from Papers where NormalizedVenueName='nature'")
		df.show()
		df = sqlContext.sql("select NormalizedVenueName from Papers")
		df.distinct().count() // res8: Long = 24004
		val res = df.distinct()
		res.rdd.map(x=>x.mkString(",")).foreach(println)
		res.rdd.saveAsTextFile("hdfs://atlas8:9000/data/output")

		df = sqlContext.sql("select FieldOfStudyName from FieldsOfStudy")
		df.show()
		df.rdd.saveAsTextFile("hdfs://atlas8:9000/data/fieldsofstudy")
		// hadoop fs -copyToLocal /data/fieldsofstudy/part-00000 .
		// grep "Computer vision" part-00000 > Mining.txt
		var df2 = sqlContext.sql("select FieldOfStudyID from FieldsOfStudy where FieldOfStudyName='Computer vision'")
		df2.show() // 01E7DD16
		val df3 = sqlContext.sql("select PaperID from PaperKeywords where fieldofstudyidmappedtokeyword='01E7DD16'")
		df3.count()

		
		
		
		
		
		
		
		
		
		
		// Load all MAG tables into SparkSQL database
		sqlContext.sql("CREATE TABLE Journals (JournalID String,JournalName String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
		sqlContext.sql("CREATE TABLE ConferenceSeries (ConferenceSeriesID String,ShortName String,FullName String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
		sqlContext.sql("CREATE TABLE ConferenceInstances (ConferenceSeriesID String,ConferenceInstanceID String,ShortName String,FullName String,Location String,OfficialConferenceURL String,ConferenceStartDate String,ConferenceEndDate String,ConferenceAbstractRegistrationDate String,ConferenceSubmissionDeadlineDate String,ConferenceNotificationDueDate String,ConferenceFinalVersionDueDate String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")

		sqlContext.sql("CREATE TABLE Affiliations (AffiliationID String,AffiliationName String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
		sqlContext.sql("CREATE TABLE Authors (AuthorID String,AuthorName String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
		sqlContext.sql("CREATE TABLE FieldsOfStudy (FieldOfStudyID String,FieldOfStudyName String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
		sqlContext.sql("CREATE TABLE FieldOfStudyHierarchy (ChildFieldOfStudyID String,ChildFieldOfStudyLevel String,ParentFieldOfStudyID String,ParentFieldOfStudyLevel String,Confidence String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
		sqlContext.sql("CREATE TABLE Journals (JournalID String,JournalName String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
		sqlContext.sql("CREATE TABLE Papers (PaperID String,OriginalPaperTitle String,NormalizedPaperTitle String,PaperPublishYear String,PaperPublishDate String,PaperDocumentObjectIdentifier String,OriginalVenueName String,NormalizedVenueName String,JournalIDMappedToVenueName String,ConferenceSeriesIDMappedToVenueName String,PaperRank String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
		sqlContext.sql("CREATE TABLE PaperAuthorAffiliations (PaperID String,AuthorID String,AffiliationID String,OriginalAffiliationName String,NormalizedAffiliationName String,AuthorSequenceNumber String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
		sqlContext.sql("CREATE TABLE PaperKeywords (PaperID String,KeywordName String,FieldOfStudyIDMappedToKeyword String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
		sqlContext.sql("CREATE TABLE PaperReferences (PaperID String,PaperReferenceID String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
		sqlContext.sql("CREATE TABLE PaperUrls (PaperID String,URL String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
		
		sqlContext.sql("LOAD DATA INPATH '/data/Journals.txt' OVERWRITE INTO TABLE Journals")
		sqlContext.sql("LOAD DATA INPATH '/data/Conferences.txt' OVERWRITE INTO TABLE ConferenceSeries")
		sqlContext.sql("LOAD DATA INPATH '/data/ConferenceInstances.txt' OVERWRITE INTO TABLE ConferenceInstances")

		sqlContext.sql("LOAD DATA INPATH '/data/Affiliations.txt' OVERWRITE INTO TABLE Affiliations")
		sqlContext.sql("LOAD DATA INPATH '/data/Authors.txt' OVERWRITE INTO TABLE Authors")
		sqlContext.sql("LOAD DATA INPATH '/data/FieldsOfStudy.txt' OVERWRITE INTO TABLE FieldsOfStudy")
		sqlContext.sql("LOAD DATA INPATH '/data/FieldOfStudyHierarchy.txt' OVERWRITE INTO TABLE FieldOfStudyHierarchy")
		sqlContext.sql("LOAD DATA INPATH '/data/Journals.txt' OVERWRITE INTO TABLE Journals")
		sqlContext.sql("LOAD DATA INPATH '/data/Papers.txt' OVERWRITE INTO TABLE Papers")
		sqlContext.sql("LOAD DATA INPATH '/data/PaperAuthorAffiliations.txt' OVERWRITE INTO TABLE PaperAuthorAffiliations")
		sqlContext.sql("LOAD DATA INPATH '/data/PaperKeywords.txt' OVERWRITE INTO TABLE PaperKeywords")
		sqlContext.sql("LOAD DATA INPATH '/data/PaperReferences.txt' OVERWRITE INTO TABLE PaperReferences")
		sqlContext.sql("LOAD DATA INPATH '/data/PaperUrls.txt' OVERWRITE INTO TABLE PaperUrls")
		
		
		sqlContext.sql("select JournalName from Journals").distinct().rdd.saveAsTextFile("hdfs://atlas8:9000/data/outputjnj")
		sqlContext.sql("select FullName from ConferenceSeries").distinct().rdd.saveAsTextFile("hdfs://atlas8:9000/data/outputfncs")
		sqlContext.sql("select ShortName from ConferenceInstances").distinct().rdd.saveAsTextFile("hdfs://atlas8:9000/data/outputsnci")
		sqlContext.sql("select FullName from ConferenceInstances").distinct().rdd.saveAsTextFile("hdfs://atlas8:9000/data/outputfnci")


		
		
		
		
		
		
		
		
		
		
		// Search for computer science publications


//	 sqlContext.sql("USE mydb")

		
		
		
		
		// Search for exact fields of study
		df2 = sqlContext.sql("select FieldOfStudyID from FieldsOfStudy where FieldOfStudyName='Data mining'")
		df2.show() // 0765A2E4
		val df4 = sqlContext.sql("select PaperID from PaperKeywords where fieldofstudyidmappedtokeyword='0765A2E4'")

		df2 = sqlContext.sql("select FieldOfStudyID from FieldsOfStudy where FieldOfStudyName='Algorithm'")
		df2.show() // 00AE2819
		val df5 = sqlContext.sql("select PaperID from PaperKeywords where fieldofstudyidmappedtokeyword='00AE2819'")

		df2 = sqlContext.sql("select FieldOfStudyID from FieldsOfStudy where FieldOfStudyName='Artificial intelligence'")
		df2.show() // 093C4716
		val df6 = sqlContext.sql("select PaperID from PaperKeywords where fieldofstudyidmappedtokeyword='093C4716'")

		df2 = sqlContext.sql("select FieldOfStudyID from FieldsOfStudy where FieldOfStudyName='Computer network'")
		df2.show() // 01DCF91B
		val df7 = sqlContext.sql("select PaperID from PaperKeywords where fieldofstudyidmappedtokeyword='01DCF91B'")

		df2 = sqlContext.sql("select FieldOfStudyID from FieldsOfStudy where FieldOfStudyName='Machine learning'")
		df2.show() // 0724DFBA
		val df8 = sqlContext.sql("select PaperID from PaperKeywords where fieldofstudyidmappedtokeyword='0724DFBA'")

		df2 = sqlContext.sql("select FieldOfStudyID from FieldsOfStudy where FieldOfStudyName='Simulation'")
		df2.show() // 02A1BFD4
		val df9 = sqlContext.sql("select PaperID from PaperKeywords where fieldofstudyidmappedtokeyword='02A1BFD4'")

		df2 = sqlContext.sql("select FieldOfStudyID from FieldsOfStudy where FieldOfStudyName='Pattern recognition'")
		df2.show() // 0AAB07DF, 09215ADF
		val df10 = sqlContext.sql("select PaperID from PaperKeywords where fieldofstudyidmappedtokeyword='0AAB07DF'")
		val df11 = sqlContext.sql("select PaperID from PaperKeywords where fieldofstudyidmappedtokeyword='09215ADF'")
		
		df2 = sqlContext.sql("select FieldOfStudyID from FieldsOfStudy where FieldOfStudyName='Real-time computing'")
		df2.show() // 04BB9B33
		val df12 = sqlContext.sql("select PaperID from PaperKeywords where fieldofstudyidmappedtokeyword='04BB9B33'")
		
		df2 = sqlContext.sql("select FieldOfStudyID from FieldsOfStudy where FieldOfStudyName='Computer hardware'")
		df2.show() // 008F4943
		val df13 = sqlContext.sql("select PaperID from PaperKeywords where fieldofstudyidmappedtokeyword='008F4943'")
		
		val dffin = df13.unionAll(df12).unionAll(df11).unionAll(df10).unionAll(df9).unionAll(df8).unionAll(df7).unionAll(df6).unionAll(df5).unionAll(df4).unionAll(df3)				
		dffin.count() // 1254879
		dffin.rdd.saveAsTextFile("/data/filteredcspaperids")
		
		
		
    dffin.registerTempTable("papersdf") 
		sqlContext.sql("CREATE TABLE FilteredPapersID (PaperID String) as select * from papersdf")
		sqlContext.sql("CREATE TABLE FilteredPapersID (PaperID String)")
    sqlContext.sql("LOAD DATA INPATH '/data/filteredcspaperids' OVERWRITE INTO TABLE FilteredPapersID")
		
    
    
    
    
    
    
    

		// Search for approximate fields of study

		sqlContext.sql("select FullName from ConferenceInstances where FullName rlike '^.*Artificial.*Intelligence.*$' ").count()
		sqlContext.sql("select FullName from ConferenceInstances where FullName rlike '^.*Artificial.*Intelligence.*$' ").show()
		sqlContext.sql("select FullName from ConferenceInstances where ShortName rlike '^.*AAAI.*$' ").show()
		
		
		var res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*AAAI.*$' ")
		var res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Aa]rtificial.*[Ii]ntelligence.*$' ")
		var res3 = res1.unionAll(res2).distinct() // res3.count() : 541

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*AAMAS.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Aa]gents.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) // res3.count() : 702

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*ACMMM.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Mm]ultimedia.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) // res3.count() : 1383

		res1 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*Statistic.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Mm]edicine.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) // res3.count() : 1667

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*AMCIS.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Ii]nfo.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) // res3.count() : 10539

		res3.count()
		res3.rdd.saveAsTextFile("/data/res31")
		res3.write.format("com.databricks.spark.csv").save("/home/achivuku/output/resrdds/res31.csv")
		
		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*AQIS.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Qq]uantum.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) // res3.count() : 10573

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*ASONAM.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Nn]etwork.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) // res3.count() : 13873

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*BioMed.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Bb]io.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) // res3.count() : 15554

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*CASoN.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Cc]omput.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) // res3.count() : 29944

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*CAV.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Vv]erifi.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) // res3.count() : 18499

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*CCC.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Cc]omplex.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) // res3.count() : 18979

		res1 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Cc]luster.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Cc]loud.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) // res3.count() : 19877

		res1 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Cc]ommunicat.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Ss]ecur.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) // res3.count() : 21738

		res1 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Kk]no.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*ACM.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) // res3.count() : 23049

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*COGSCI.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[cC]ognit.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) // res3.count() : 23672

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*CONCUR.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[cC]oncur.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) // res3.count() : 23725

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*CoopIS.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[cC]oop.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*COSIT.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Ss]pat.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*CP.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Pp]rogram.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*Crypto.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[cC]ryptology.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*CSF.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Ss]ecurity.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*CVPR.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[pP]attern.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*DASFAA.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[dD]ata.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*ECCV.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[vV]ision.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*ECIS.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*ECML.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*ECML.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Mm]achine.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*EDBT.*$' ")
		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*FOCS.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*FUZZ.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Ff]uzzy.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*GECCO.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Gg]ene.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Ee]volution.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Ff]ormal.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*HRI.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Rr]obo.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*Humanoids.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Hh]uman.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*Humanoids.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Hh]uman.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*Hypertext.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Hh]yper.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*ICALP.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Aa]utomat.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*ICAPS.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Pp]lan.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*ICCV.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*IEEE.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*ICDE.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*ICDM.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*ICFP.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Ff]unction.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*ICIEA.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Ee]lectronic.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*ICIS.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*ICML.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*ICRA.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*ICSR.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*ICSOC.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Ss]ervice.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*CEC.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*CIBCB.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*IJCNN.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*IROS.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*ICSOC.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Nn]eur.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*ISD.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*ISIT.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*KDD.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Kk]no.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*KR.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Rr]eason.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*LICS.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Ll]ogic.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*NIPS.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*PACIS.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*PAKDD.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*Data.*Mining.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*PODS.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*SIG.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*POPL.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Ll]anguage.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*PPSN.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Pp]arallel.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*QCMC.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Mm]easure.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*Qcrypt.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Cc]rypto.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*QIP.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Pp]rocess.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*QIPC.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*RiTA.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*RecSys.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Rr]ecommend.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*MAN.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Ii]nteract.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*S.*P.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Pp]riv.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*SDM.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*SIGIR.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*SIGKDD.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Kk]nowledge.*[Dd]iscovery.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*SIGMOD.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Dd]ata.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*SSTD.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*STACS.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*STOC.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Tt]heory.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*TACAS.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Aa]lgorithm.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*TARK.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Rr]ation.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*TLCA.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Ll]ambda.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*TQC.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*UAI.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*TLCA.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*VLDB.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*WCCI.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Ii]ntelligence.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*WISE.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*[Ww]eb.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*WWW.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*Problem.*Solving.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res1 = sqlContext.sql("select * from ConferenceInstances where ShortName rlike '^.*MFCS.*$' ")
		res2 = sqlContext.sql("select * from ConferenceInstances where FullName rlike '^.*Computer.*Science.*$' ")
		res3 = res3.unionAll(res1.unionAll(res2).distinct()) 

		res3.count()

    
		
		
		
		
		
		// Split and Join MAG database to create CS database		
		  
    sqlContext.sql("CREATE TABLE Papersaa (PaperID String,OriginalPaperTitle String,NormalizedPaperTitle String,PaperPublishYear String,PaperPublishDate String,PaperDocumentObjectIdentifier String,OriginalVenueName String,NormalizedVenueName String,JournalIDMappedToVenueName String,ConferenceSeriesIDMappedToVenueName String,PaperRank String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
    sqlContext.sql("CREATE TABLE Papersab (PaperID String,OriginalPaperTitle String,NormalizedPaperTitle String,PaperPublishYear String,PaperPublishDate String,PaperDocumentObjectIdentifier String,OriginalVenueName String,NormalizedVenueName String,JournalIDMappedToVenueName String,ConferenceSeriesIDMappedToVenueName String,PaperRank String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
    sqlContext.sql("LOAD DATA INPATH '/data/PapersSplits/splitpapersaa' OVERWRITE INTO TABLE Papersaa")
    sqlContext.sql("LOAD DATA INPATH '/data/PapersSplits/splitpapersab' OVERWRITE INTO TABLE Papersab")
    
    sqlContext.sql("CREATE TABLE PaperAuthorAffiliationsaa (PaperID String,AuthorID String,AffiliationID String,OriginalAffiliationName String,NormalizedAffiliationName String,AuthorSequenceNumber String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
    sqlContext.sql("CREATE TABLE PaperAuthorAffiliationsab (PaperID String,AuthorID String,AffiliationID String,OriginalAffiliationName String,NormalizedAffiliationName String,AuthorSequenceNumber String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
    sqlContext.sql("CREATE TABLE PaperAuthorAffiliationsac (PaperID String,AuthorID String,AffiliationID String,OriginalAffiliationName String,NormalizedAffiliationName String,AuthorSequenceNumber String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
    sqlContext.sql("CREATE TABLE PaperAuthorAffiliationsad (PaperID String,AuthorID String,AffiliationID String,OriginalAffiliationName String,NormalizedAffiliationName String,AuthorSequenceNumber String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
    sqlContext.sql("LOAD DATA INPATH '/data/PaperAuthorAffiliationsSplits/splitpapersauthorsaa' OVERWRITE INTO TABLE PaperAuthorAffiliationsaa")
    sqlContext.sql("LOAD DATA INPATH '/data/PaperAuthorAffiliationsSplits/splitpapersauthorsab' OVERWRITE INTO TABLE PaperAuthorAffiliationsab")
    sqlContext.sql("LOAD DATA INPATH '/data/PaperAuthorAffiliationsSplits/splitpapersauthorsac' OVERWRITE INTO TABLE PaperAuthorAffiliationsac")
    sqlContext.sql("LOAD DATA INPATH '/data/PaperAuthorAffiliationsSplits/splitpapersauthorsad' OVERWRITE INTO TABLE PaperAuthorAffiliationsad")

    sqlContext.sql("CREATE TABLE PaperReferencesaa (PaperID String,PaperReferenceID String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
    sqlContext.sql("CREATE TABLE PaperReferencesab (PaperID String,PaperReferenceID String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
    sqlContext.sql("CREATE TABLE PaperReferencesac (PaperID String,PaperReferenceID String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
    sqlContext.sql("CREATE TABLE PaperReferencesad (PaperID String,PaperReferenceID String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
    sqlContext.sql("CREATE TABLE PaperReferencesae (PaperID String,PaperReferenceID String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
    sqlContext.sql("CREATE TABLE PaperReferencesaf (PaperID String,PaperReferenceID String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
    sqlContext.sql("LOAD DATA INPATH '/data/PaperReferencesSplits/splitpaperrefsaa' OVERWRITE INTO TABLE PaperReferencesaa")
    sqlContext.sql("LOAD DATA INPATH '/data/PaperReferencesSplits/splitpaperrefsab' OVERWRITE INTO TABLE PaperReferencesab")
    sqlContext.sql("LOAD DATA INPATH '/data/PaperReferencesSplits/splitpaperrefsac' OVERWRITE INTO TABLE PaperReferencesac")
    sqlContext.sql("LOAD DATA INPATH '/data/PaperReferencesSplits/splitpaperrefsad' OVERWRITE INTO TABLE PaperReferencesad")
    sqlContext.sql("LOAD DATA INPATH '/data/PaperReferencesSplits/splitpaperrefsae' OVERWRITE INTO TABLE PaperReferencesae")
    sqlContext.sql("LOAD DATA INPATH '/data/PaperReferencesSplits/splitpaperrefsaf' OVERWRITE INTO TABLE PaperReferencesaf")
    
    sqlContext.sql("CREATE TABLE PaperKeywordsaa (PaperID String,KeywordName String,FieldOfStudyIDMappedToKeyword String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
    sqlContext.sql("CREATE TABLE PaperKeywordsab (PaperID String,KeywordName String,FieldOfStudyIDMappedToKeyword String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
    sqlContext.sql("LOAD DATA INPATH '/data/PaperKeywordsSplits/splitpaperkeysaa' OVERWRITE INTO TABLE PaperKeywordsaa")
    sqlContext.sql("LOAD DATA INPATH '/data/PaperKeywordsSplits/splitpaperkeysab' OVERWRITE INTO TABLE PaperKeywordsab")
    
    
    sqlContext.sql("CREATE TABLE CSPaperAuthorAffiliations (PaperID String,AuthorID String,AffiliationID String,OriginalAffiliationName String,NormalizedAffiliationName String,AuthorSequenceNumber String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
    sqlContext.sql("LOAD DATA INPATH '/data/PaperAuthorAffiliationsSplits/splitpapersauthorsaa' OVERWRITE INTO TABLE PaperAuthorAffiliationsaa")
    
    
    sqlContext.sql("CREATE TABLE Authorsaa (AuthorID String,AuthorName String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
    sqlContext.sql("CREATE TABLE Authorsab (AuthorID String,AuthorName String) ROW FORMAT delimited FIELDS TERMINATED BY '\t' STORED AS textfile")
    sqlContext.sql("LOAD DATA INPATH '/data/AuthorsSplits/splitauthorsaa' OVERWRITE INTO TABLE Authorsaa")
    sqlContext.sql("LOAD DATA INPATH '/data/AuthorsSplits/splitauthorsab' OVERWRITE INTO TABLE Authorsab")

           
		
    val papersids =  sqlContext.sql("select PaperID from FilteredPapersID")    
    
    val papersaa = sqlContext.sql("select * from Papersaa LIMIT 100000000")
    val papersab = sqlContext.sql("select * from Papersab LIMIT 100000000")
    
    val paperauthoraffiliationsaa = sqlContext.sql("select * from PaperAuthorAffiliationsaa LIMIT 100000000")
    val paperauthoraffiliationsab = sqlContext.sql("select * from PaperAuthorAffiliationsab LIMIT 100000000")
    val paperauthoraffiliationsac = sqlContext.sql("select * from PaperAuthorAffiliationsac LIMIT 100000000")
    val paperauthoraffiliationsad = sqlContext.sql("select * from PaperAuthorAffiliationsad LIMIT 100000000")

    val paperreferencesaa = sqlContext.sql("select * from PaperReferencesaa LIMIT 100000000")
    val paperreferencesab = sqlContext.sql("select * from PaperReferencesab LIMIT 100000000")
    val paperreferencesac = sqlContext.sql("select * from PaperReferencesac LIMIT 100000000")
    val paperreferencesad = sqlContext.sql("select * from PaperReferencesad LIMIT 100000000")
    val paperreferencesae = sqlContext.sql("select * from PaperReferencesae LIMIT 100000000")
    val paperreferencesaf = sqlContext.sql("select * from PaperReferencesaf LIMIT 100000000")

    val paperkeywordsaa = sqlContext.sql("select * from PaperKeywordsaa LIMIT 100000000")
    val paperkeywordsab = sqlContext.sql("select * from PaperKeywordsab LIMIT 100000000")

    val authorsaa = sqlContext.sql("select * from Authorsaa LIMIT 100000000")
    val authorsab = sqlContext.sql("select * from Authorsab LIMIT 100000000")    
    
    papersids.join(papersaa, papersids("PaperID") === papersaa("PaperID"), "inner").drop(papersaa.col("paperid")).rdd.saveAsTextFile("hdfs://atlas8:9000/data/papersaa")
    papersids.join(papersab, papersids("PaperID") === papersab("PaperID"), "inner").drop(papersab.col("paperid")).rdd.saveAsTextFile("hdfs://atlas8:9000/data/papersab")
    
    papersids.join(paperauthoraffiliationsaa, papersids("PaperID") === paperauthoraffiliationsaa("PaperID"), "inner").drop(paperauthoraffiliationsaa.col("paperid")).rdd.saveAsTextFile("hdfs://atlas8:9000/data/paperauthoraffiliationsaa")
    papersids.join(paperauthoraffiliationsab, papersids("PaperID") === paperauthoraffiliationsab("PaperID"), "inner").drop(paperauthoraffiliationsab.col("paperid")).rdd.saveAsTextFile("hdfs://atlas8:9000/data/paperauthoraffiliationsab")
    papersids.join(paperauthoraffiliationsac, papersids("PaperID") === paperauthoraffiliationsac("PaperID"), "inner").drop(paperauthoraffiliationsac.col("paperid")).rdd.saveAsTextFile("hdfs://atlas8:9000/data/paperauthoraffiliationsac")
    papersids.join(paperauthoraffiliationsad, papersids("PaperID") === paperauthoraffiliationsad("PaperID"), "inner").drop(paperauthoraffiliationsad.col("paperid")).rdd.saveAsTextFile("hdfs://atlas8:9000/data/paperauthoraffiliationsad")
    
    papersids.join(paperreferencesaa, papersids("PaperID") === paperreferencesaa("PaperID"), "inner").drop(paperreferencesaa.col("paperid")).rdd.saveAsTextFile("hdfs://atlas8:9000/data/paperreferencesaa")
    papersids.join(paperreferencesab, papersids("PaperID") === paperreferencesab("PaperID"), "inner").drop(paperreferencesab.col("paperid")).rdd.saveAsTextFile("hdfs://atlas8:9000/data/paperreferencesab")
    papersids.join(paperreferencesac, papersids("PaperID") === paperreferencesac("PaperID"), "inner").drop(paperreferencesac.col("paperid")).rdd.saveAsTextFile("hdfs://atlas8:9000/data/paperreferencesac")
    papersids.join(paperreferencesad, papersids("PaperID") === paperreferencesad("PaperID"), "inner").drop(paperreferencesad.col("paperid")).rdd.saveAsTextFile("hdfs://atlas8:9000/data/paperreferencesad")
    papersids.join(paperreferencesae, papersids("PaperID") === paperreferencesae("PaperID"), "inner").drop(paperreferencesae.col("paperid")).rdd.saveAsTextFile("hdfs://atlas8:9000/data/paperreferencesae")
    papersids.join(paperreferencesaf, papersids("PaperID") === paperreferencesaf("PaperID"), "inner").drop(paperreferencesaf.col("paperid")).rdd.saveAsTextFile("hdfs://atlas8:9000/data/paperreferencesaf")
    
    papersids.join(paperkeywordsaa, papersids("PaperID") === paperkeywordsaa("PaperID"), "inner").drop(paperkeywordsaa.col("paperid")).rdd.saveAsTextFile("hdfs://atlas8:9000/data/paperkeywordsaa")
    papersids.join(paperkeywordsab, papersids("PaperID") === paperkeywordsab("PaperID"), "inner").drop(paperkeywordsab.col("paperid")).rdd.saveAsTextFile("hdfs://atlas8:9000/data/paperkeywordsab")
    

    val papers = sqlContext.sql("select * from Papers")
    val authors = sqlContext.sql("select * from PaperAuthorAffiliations")
    papers.count() // 126909021
    authors.count() // 337000600
//		sqlContext.sql("DROP TABLE filteredpapers")
		papersids.join(papers, papersids("PaperID") === papers("PaperID"), "inner").drop(papers.col("paperid")).registerTempTable("papersf") 
		sqlContext.sql("create table filteredpapers as select * from papersf");
		
    papersids.join(papers, papersids("PaperID") === papers("PaperID"), "inner").drop(papers.col("paperid")).rdd.saveAsTextFile("hdfs://atlas8:9000/data/filteredcspapers")
    
    papersids.join(authors, papersids("PaperID") === authors("PaperID"), "inner").drop(authors.col("paperid")).registerTempTable("authorsf") 
		sqlContext.sql("create table filteredauthors as select * from authorsf");
    // papersids.join(authors, papersids("PaperID") === authors("PaperID"), "inner").drop(authors.col("paperid")).count()
		// sqlContext.sql("drop table filteredauthors")


    
    
    
    
    
    // Create CS database 
    
    sqlContext.sql("CREATE TABLE CSPaperAuthorAffiliations (PaperID String,AuthorID String,AffiliationID String,OriginalAffiliationName String,NormalizedAffiliationName String,AuthorSequenceNumber String) ROW FORMAT delimited FIELDS TERMINATED BY ',' STORED AS textfile")
    sqlContext.sql("LOAD DATA INPATH '/data/CSPaperAuthorAffiliations.csv' OVERWRITE INTO TABLE CSPaperAuthorAffiliations")
    val cspaperauthoraffiliations = sqlContext.sql("select * from CSPaperAuthorAffiliations")
    cspaperauthoraffiliations.count() // 3863333
    val csauthorids = sqlContext.sql("select AuthorID from CSPaperAuthorAffiliations")
    csauthorids.printSchema()
    authorsaa.printSchema()
    
    csauthorids.join(authorsaa, csauthorids("AuthorID") === authorsaa("AuthorID"), "inner").drop(authorsaa.col("authorid")).rdd.saveAsTextFile("hdfs://atlas8:9000/data/authorsaa")
    csauthorids.join(authorsab, csauthorids("AuthorID") === authorsab("AuthorID"), "inner").drop(authorsab.col("authorid")).rdd.saveAsTextFile("hdfs://atlas8:9000/data/authorsab")

    
    sqlContext.sql("CREATE TABLE CSAuthors (AuthorID String,AuthorName String) ROW FORMAT delimited FIELDS TERMINATED BY ',' STORED AS textfile")
    sqlContext.sql("LOAD DATA INPATH '/data/CSAuthors.csv' OVERWRITE INTO TABLE CSAuthors")
    val csauthors = sqlContext.sql("select * from CSAuthors")
    csauthors.count() // 3863333

    sqlContext.sql("CREATE TABLE CSPapers (PaperID String,OriginalPaperTitle String,NormalizedPaperTitle String,PaperPublishYear String,PaperPublishDate String,PaperDocumentObjectIdentifier String,OriginalVenueName String,NormalizedVenueName String,JournalIDMappedToVenueName String,ConferenceSeriesIDMappedToVenueName String,PaperRank String) ROW FORMAT delimited FIELDS TERMINATED BY ',' STORED AS textfile")
    sqlContext.sql("LOAD DATA INPATH '/data/CSPapers.csv' OVERWRITE INTO TABLE CSPapers")
    val cspapers = sqlContext.sql("select * from CSPapers")
    cspapers.count() // 1254879
    
    sqlContext.sql("CREATE TABLE CSPaperKeywords (PaperID String,KeywordName String,FieldOfStudyIDMappedToKeyword String) ROW FORMAT delimited FIELDS TERMINATED BY ',' STORED AS textfile")
    sqlContext.sql("LOAD DATA INPATH '/data/CSPaperKeywords.csv' OVERWRITE INTO TABLE CSPaperKeywords")
    val cspaperkeywords = sqlContext.sql("select * from CSPaperKeywords")
    cspaperkeywords.count() // 10760875
    
    sqlContext.sql("CREATE TABLE CSPaperReferences (PaperID String,PaperReferenceID String) ROW FORMAT delimited FIELDS TERMINATED BY ',' STORED AS textfile")
    sqlContext.sql("LOAD DATA INPATH '/data/CSPaperReferences.csv' OVERWRITE INTO TABLE CSPaperReferences")
    val cspaperreferences = sqlContext.sql("select * from CSPaperReferences")
    cspaperreferences.count() // 8179711
   
    
		
	}
		

   
}



/*

Shell Commands

sed 's/\[//g' merged-file > merged-file2
sed 's/\]//g' merged-file2 > merged-file3
cp merged-file3 merged-file

HiveQL

Check Missing Values :

select count(*) from Affiliations;
select year(from_unixtime(unix_timestamp(ConferenceEndDate,'yyyy/mm/dd'), 'yyyy-mm-dd')) from ConferenceInstances;
describe ConferenceInstances;
select ConferenceStartDate,ConferenceEndDate,ConferenceAbstractRegistrationDate, ConferenceSubmissionDeadlineDate, ConferenceNotificationDueDate, ConferenceFinalVersionDueDate from ConferenceInstances where ConferenceStartDate=null or ConferenceEndDate=null or ConferenceAbstractRegistrationDate=null or ConferenceSubmissionDeadlineDate=null or ConferenceNotificationDueDate=null or ConferenceFinalVersionDueDate=null;
select NormalizedPaperTitle,NormalizedVenueName from Papers limit 10;
select * from Papers where NormalizedVenueName='nature';


HDFS Shell Commands :

hadoop fs -ls /data
hadoop fs -copyToLocal /data/output .
cd ~/output; cat * > merged-file

hadoop fs -copyToLocal /data/outputjnj .
hadoop fs -copyToLocal /data/outputsncs .
hadoop fs -copyToLocal /data/outputfncs .
hadoop fs -copyToLocal /data/outputsnci .
hadoop fs -copyToLocal /data/outputfnci .
hadoop fs -copyFromLocal /home/achivuku/Documents/MicrosoftAcademicGraph/ /data

cd ~/output/filteredcspublications
hadoop fs -copyToLocal /data/filteredcspapersaa .
hadoop fs -copyToLocal /data/filteredcspapersab .

hadoop fs -copyToLocal /data/paperauthoraffiliationsaa .
hadoop fs -copyToLocal /data/paperauthoraffiliationsab .
hadoop fs -copyToLocal /data/paperauthoraffiliationsac .
hadoop fs -copyToLocal /data/paperauthoraffiliationsad .

hadoop fs -copyToLocal /data/paperreferencesaa .
hadoop fs -copyToLocal /data/paperreferencesab .
hadoop fs -copyToLocal /data/paperreferencesac .
hadoop fs -copyToLocal /data/paperreferencesad .
hadoop fs -copyToLocal /data/paperreferencesae .
hadoop fs -copyToLocal /data/paperreferencesaf .

hadoop fs -copyToLocal /data/paperkeywordsaa .
hadoop fs -copyToLocal /data/paperkeywordsab .

hadoop fs -copyToLocal /data/authorsaa .
hadoop fs -copyToLocal /data/authorsab .

hadoop fs -copyFromLocal /home/achivuku/output/filteredcspublications/cspaperauthoraffiliations/merged-file /data/CSPaperAuthorAffiliations.csv
hadoop fs -copyFromLocal /home/achivuku/output/filteredcspublications/csauthors/merged-file /data/CSAuthors.csv
hadoop fs -copyFromLocal /home/achivuku/output/filteredcspublications/cspapers/merged-file /data/CSPapers.csv
hadoop fs -copyFromLocal /home/achivuku/output/filteredcspublications/cspaperkeywords/merged-file /data/CSPaperKeywords.csv
hadoop fs -copyFromLocal /home/achivuku/output/filteredcspublications/cspaperreferences/merged-file /data/CSPaperReferences.csv

Spark Shell Commands :
spark-shell -Dspark.executor.memory=50g
spark-shell  --executor-memory 16G

Python Shell Commands :

l = open("/home/aneesh/Documents/TopConferenceList.txt").readlines()
l2 = map(str.strip, l[1::2][0:-1])
l2 = map(str.strip, l[0::2][1:-1])

Query Conferences :

['AAAI', 'AAMAS', 'ACMMM', 'AIIM', 'AISTATS', 'AMCIS', 'AQIS', 'ASONAM', 'BioMed', 'CASoN', 'CAV', 'CCC', 'CCGRID', 'CCS', 'CIKM', 'COGSCI', 'CONCUR', 'CoopIS', 'COSIT', 'CP', 'Crypto', 'CSF', 'CVPR', 'DASFAA', 'ECAI', 'ECCV', 'ECIS', 'ECML', 'ECML-PKDD', 'EDBT', 'FME', 'FOCS', 'FUZZ-IEEE', 'GECCO', 'HRI', 'Humanoids', 'Hypertext', 'ICALP', 'ICAPS', 'ICCV', 'ICDE', 'ICDM', 'ICFP', 'ICIEA', 'ICIS', 'ICML', 'ICRA', 'ICSOC', 'ICSR', 'IEEE CEC', 'IEEE CIBCB', 'IJCAI', 'IJCNN', 'IJCNN', 'IROS', 'ISD', 'ISIT', 'JELIA', 'KDD', 'KR', 'LICS', 'LICS', 'MFCS', 'NIPS', 'PACIS', 'PAKDD', 'PODS', 'POPL', 'PPSN', 'QCMC', 'Qcrypt', 'QIP', 'QIPC', 'RecSys', 'RiTA', 'RO-MAN', 'S&P', 'SDM', 'SIGIR', 'SIGKDD', 'SIGMOD', 'SSTD', 'STACS', 'STOC', 'TACAS', 'TARK', 'TLCA', 'TQC', 'UAI', 'VLDB', 'WCCI', 'WISE', 'WWW']

['National Conference of the American Association for Artificial Intelligence', 'International Conference on Autonomous Agents', 'ACM Multimedia', 'Artificial Intelligence in Medicine', 'International Conference on Artificial Intelligence and Statistics', 'Americas Conference on Information Systems', 'Asian Quantum Information Science Conference', 'IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining', 'IASTED International Conference on Biomedical Engineering', 'International Conference on Computational Aspects of Social Networks', 'Computer Aided Verification', 'Computational Complexity Conference', 'IEEE/ACM International Symposium on Cluster, Cloud and Grid Computing', 'ACM Conference on Computer and Communications Security', 'ACM International Conference on Information and Knowledge Management', 'Annual Conference of the Cognitive Science Society', 'International Conference on Concurrency Theory', 'International Conference on Cooperative Information Systems', 'Conference on Spatial Information Theory', 'International Conference on Principles and Practice of Constraint Programming', 'Advances in Cryptology', 'IEEE Computer Security Foundations Symposium', 'IEEE Conference on Computer Vision and Pattern Recognition', 'Database Systems for Advanced Applications', 'European Conference on Artificial Intelligence', 'European Conference on Computer Vision', 'European Conference on Information Systems', 'European Conference on Machine Learning', 'European Conference on Principles and Practice of Knowledge Discovery in Databases', 'Extending Database Technology', 'Formal Methods Europe', 'IEEE Symposium on Foundations of Computer Science', 'IEEE International Conference on Fuzzy Systems', 'Genetic and Evolutionary Computations', 'ACM/IEEE International Conference on Human Robot Interaction', 'IEEE-RAS International Conference on Humanoid Robots (Humanoids)', 'ACM Conference on Hypertext and Hypermedia', 'International Colloquium on Automata Languages and Programming', 'International Conference on Automated Planning and Scheduling', 'IEEE International Conference on Computer Vision', 'International Conference on Data Engineering', 'IEEE International Conference on Data Mining', 'International Conference on Functional Programming', 'IEEE Conference in Industrial Electronics and Applications', 'International Conference on Information Systems', 'International Conference on Machine Learning', 'IEEE International Conference on Robotics & Automation', 'International Conference on Service Oriented Computing', 'International Conference on Social Robotics', 'IEEE Congress on Evolutionary Computation', 'IEEE Symposium on Computational Intelligence in Bioinformatics and Computational Biology', 'International Joint Conference on Artificial Intelligence', 'IEEE International Joint Conference on Neural Networks', 'International Joint Conference on Neural Networks', 'IEEE/RSJ International Conference on Intelligent Robots and Systems', 'International Conference on Information Systems Development', 'IEEE International Symposium on Information Theory', 'Logics in Artificial Intelligence, European Conference', 'Knowledge Discovery & Data Mining', 'International Conference on the Principles of Knowledge Representation and Reasoning', 'IEEE Symposium on Logic in Computer Science', 'Logic in Computer Science', 'International Symposium on Mathematical Foundations of Computer Science', 'Advances in Neural Information Processing Systems', 'Pacific-Asia Conference on Information Systems', 'Pacific-Asia Conference on Knowledge Discovery and Data Mining', 'ACM SIGMOD-SIGACT-SIGART Conference on Principles of Database Systems', 'ACM-SIGACT Symposium on Principles of Programming Languages', 'Parallel Problem Solving from Nature', 'International Conference on Quantum Communication, Measurement and Computing', 'International Conference on Quantum Cryptography', 'Workshop on Quantum Information Processing', 'International Conference on Quantum Information Processing and Communication', 'ACM International Conference on Recommender Systems', 'International conference on Robot Inteligence Technology and applications', 'International Symposium on Robot and Human Interactive Communication', 'IEEE Symposium on Security and Privacy', 'SIAM International Conference on Data Mining', 'ACM International Conference on Research and Development in Information Retrieval', 'ACM International Conference on Knowledge Discovery and Data Mining', 'ACM Special Interest Group on Management of Data Conference', 'International Symposium on Spatial Databases', 'Symposium on Theoretical Aspects of Computer Science', 'ACM Symposium on Theory of Computing', 'Tools and Algorithms for Construction and Analysis of Systems', 'Theoretical Aspects of Rationality and Knowledge', 'International Conference on Typed Lambda Calculus and its Applications', 'Theory of Quantum Computation, Communication and Cryptography', 'Conference in Uncertainty in Artificial Intelligence', 'International Conference on Very Large Databases', 'IEEE World Congress on Computational Intelligence', 'International Conference on Web Information Systems Engineering', 'International World Wide Web Conference']

Data Schema
https://academic.microsoft.com/
https://www.quora.com/Where-can-I-obtain-a-schema-description-of-Microsoft-Academic-Graph-data
https://academicgraph.blob.core.windows.net/graph-2015-11-06/index.html

Query Fields
Conferences, ConferenceInstances : Short name (abbreviation), Full name, 
Papers : Normalized paper title, Normalized venue name
Journals : Journal name


Spark-submit configuration :
spark-submit --class "DDL" --master yarn --deploy-mode cluster --driver-memory 16G --executor-memory 16G --num-executors 14 --executor-cores 7 ~/data-loaders_2.11-1.0.jar
ssh achivuku@atlas8
scp ./target/scala-2.11/data-loaders_2.11-1.0.jar achivuku@atlas8:~

spark-shell --packages com.databricks:spark-csv_2.11:1.4.0
spark-shell  --executor-memory 16G

Search only Field of study name and join on Field of study ID, Paper ID, Conference series ID, Journal ID
In Field of study name check only exact match for Algorithm, Computer Network, Artificial Intelligence, Machine Learning, Simulation, Pattern Recognition, Real-Time Computing, Computer Hardware, Data Mining, Computer Vision. 
We expect to find 1lakh to 2lakh papers over 11 years. 
After filtering above for paper ids and removing duplicates we can join the paper ids with papers, authors, conferences to get the RDDs as coevolving networks
This way we can run SparkSQL queries on only one node to get relevant results

No field of study in 2016KDDCupSelectedPapers. Papers span for 4 years and 6 conferences only.  
3677 papers in 2016KDDCupSelectedPapers
148228 papers with keyword "Data mining"


spark-submit --class "DDL" --master yarn --deploy-mode cluster --driver-memory 16G --executor-memory 16G --num-executors 14 --executor-cores 7 ~/data-loaders_2.11-1.0.jar
SparkSQL is not working in cluster mode due to Hive dependency
Out of memory when trying to save large rdds from conference instances
Need admin support to load the database into a database server with/out hadoop
Need to get correct sqlContext, Check correct sql statements in cluster mode, use hivecontext, 

Spark API
Dataframes
http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.sql.DataFrame
SparkSQL
http://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.sql.functions$
https://spark.apache.org/docs/1.6.0/api/java/org/apache/spark/sql/Column.html
Academic Search API
https://www.microsoft.com/cognitive-services/en-us/academic-knowledge-api/documentation/overview
https://dev.projectoxford.ai/docs/services/56332331778daf02acc0a50b/operations/56332331778daf06340c9666 

Data Benchmarks :

337000600 records in PaperAuthorAffiliations table
126909021 records in Papers table
12 lakh to 12 lakh join is working with 33123 matches
120 lakh to 12 lakh join is working with 338051 matches
500 lakh to 12 lakh join is working with 999387 matches
1000 lakh or 10 crore to 12 lakh join is working with 1375042 matches
2000 lakh or 20 crore to 12 lakh join is working with 2121538 matches
100000000 records matched are stored to disk in 819.418402 or 15mins
So total runtime expected to be ~3hours

The loops over joins must be executed on following tables 
Papers.txt, PaperAuthorAffiliations.txt, PaperReferences.txt, PaperKeywords.txt, Authors.txt, in GBs
ConferenceInstances.txt, FieldOfStudyHierarchy.txt, FieldsOfStudy.txt, in MBs
Journals.txt, Affiliations.txt, Conferences.txt, in KBs

wc -l PaperReferences.txt
wc -l PaperKeywords.txt
wc -l Authors.txt

split -l 100000000 Papers.txt splitpapers
split -l 100000000 PaperAuthorAffiliations.txt splitpapersauthors
split -l 100000000 PaperReferences.txt splitpaperrefs
split -l 100000000 PaperKeywords.txt splitpaperkeys
split -l 100000000 Authors.txt splitauthors

After getting all paper ids, do inner join with papers table and journals table and authors table and conferences table to get final list of papers, authors and citations with normalized venue name and journal name
Need to join 12lakhs table with 12 crores table. OS exceptions seen if loading all 12 crores records. 
Must focus on better input/output before and after join. As workaround, need to partition the large table into small tables and aggregate results inside a parallel loop. To partition execute where and order by clause on timestamp. 
Input/output must be partitoned by timestamp and field of study. Alternate is to add a ROW_NUMBER() or RANK() to dataset and then SELECT the desired rows. 
For large joins in RDDs, SparkSQL uses broadcast variables after hive setup. Partitioning Distributed rdd requires hive. 


 */


/*
 
 Field of Study "Data" :
 
 [General Data Protection Regulation]
[Investigative Data Warehouse]
[Common Data Link]
[Database transaction]
[Torpedo Data Computer]
[Data structure]
[Data Integrity Field]
[Small Molecule Pathway Database]
[Data deficient]
[Data flow diagram]
[Data pre-processing]
[Data segment]
[Data compression]
[Conserved Domain Database]
[Meta Data Services]
[Data remanence]
[Data breach]
[Evolution-Data Optimized]
[Data transformation]
[Database design]
[User Datagram Protocol]
[Data definition language]
[Database administrator]
[Data synchronization]
[Datalog]
[National Hydrography Dataset]
[E. Coli Metabolome Database]
[Data center services]
[Datagram Transport Layer Security]
[OPC Data Access]
[Data architecture]
[Data scraping]
[Common Source Data Base]
[Data stream clustering]
[Data system]
[Service Data Objects]
[Unstructured Supplementary Service Data]
[Data Transformation Services]
[Datarâ€“Mathews method for real option valuation]
[Data quality]
[Key Sequenced Data Set]
[International Data Encryption Algorithm]
[Relative Record Data Set]
[Database theory]
[BioModels Database]
[Data file]
[Data compaction]
[Database server]
[Data scrubbing]
[Data model]
[Common Data Representation]
[Data]
[Database-centric architecture]
[Data governance]
[Data strobe encoding]
[Data]
[Data differencing]
[Database model]
[Data envelopment analysis]
[Joint Probabilistic Data Association Filter]
[Data binding]
[Data warehouse]
[Data grid]
[Data rate units]
[Fiber Distributed Data Interface]
[Data manipulation language]
[Infrared Data Association]
[Data hierarchy]
[Dataflow architecture]
[Data model]
[Database search engine]
[Open Data Protocol]
[Data transfer object]
[Datakit]
[Data transmission]
[Data cluster]
[Java Data Objects]
[Data profiling]
[Data stream mining]
[Generic Model Organism Database]
[Nursing Minimum Data Set]
[Hierarchical Data Format]
[SABIO-Reaction Kinetics Database]
[Programming with Big Data in R]
[High-Level Data Link Control]
[Database testing]
[Data access layer]
[Data Link Control]
[Data mining]
[EM Data Bank]
[Investigations in Numbers, Data, and Space]
[Data-Link Switching]
[Data verification]
[Data structure alignment]
[National Lidar Dataset]
[Data matrix]
[Datagram]
[Data at Rest]
[Data field]
[Database]
[Data aggregator]
[Data acquisition]
[Data analysis]
[Protein Data Bank (RCSB PDB)]
[Data-flow analysis]
[Data element definition]
[Data mapper pattern]
[Data loss]
[High-Speed Circuit-Switched Data]
[Data cube]
[Data center]
[OPC Historical Data Access]
[Global Data Synchronization Network]
[Database catalog]
[National Elevation Dataset]
[Open Database Connectivity]
[National Snow and Ice Data Center]
[Data access object]
[Data set]
[Extended Data Services]
[Data custodian]
[Data dictionary]
[Data service unit]
[Data independence]
[Data conversion]
[Database schema]
[Data-intensive computing]
[Data administration]
[Data point]
[Data parallelism]
[Database tuning]
[Database normalization]
[Explicit Data Graph Execution]
[Data exchange]
[Data-driven testing]
[Data visualization]
[Data signaling rate]
[Tactical Data Link]
[Data curation]
[Dataflow]
[Data logger]
[Data efficiency]
[Data truncation]
[Inorganic Crystal Structure Database]
[Data mapping]
[Data assimilation]
[Data modeling]
[Data striping]
[Data buffer]
[Data domain]
[JPL Small-Body Database]
[Secure Data Aggregation in WSN]
[Data Execution Prevention]
[Federal Reserve Economic Data]
[Database storage structures]
[Dynamic Data Exchange]
[Data virtualization]
[Data consistency]
[Data Defined Storage]
[Data as a service]
[United States Department of Energy International Energy Storage Database]
[Data processing system]
[Data processing]
[IBM 2321 Data Cell]
[Data management]
[Data redundancy]
[Remote Database Access]
[Ethernet Global Data Protocol]
[Data element]
[Data proliferation]
[Data security]
[Data deduplication]
[Data integration]
[Data compression ratio]
[National Data Repository]
[Data diffusion machine]
[Data retrieval]
[Data transformation]
[Data control language]
[Data Protection API]
[VHF Data Link]
[Data integrity]
[Data cleansing]
[Data set]
[Database index]
[Physical Data Flow]
[Data Authentication Algorithm]
[Data access]
[FlyBase : A Database of Drosophila Genes & Genomes]
[External Data Representation]
[Data link]
[Data Protection Act 1998]
[Human Metabolome Database]
[Data dredging]
[Radio Data System]
[Data validation]
[Data monitoring committee]
[Data collection]
[Data migration]
[Data reduction]
[Data Web]
[Data, context and interaction]
[SRTM Water Body Data]
[Protein Data Bank]
[Data link layer]
[Data circuit-terminating equipment]
[Circuit Switched Data]
[Disk Data Format]
[European Data Relay System]
[Data management plan]
[Standard for Exchange of Non-clinical Data]
[Data terminal equipment]
[Data Protection Directive]
[Data Reference Model]
[Data type]
[Transporter Classification Database]
[Central Air Data Computer]
[Entry Sequenced Data Set]
[Data recovery]


 
 
 Field of study "Algorithm" :
 
 [Tiny Encryption Algorithm]
[Solitaire Cryptographic Algorithm]
[Algorithm design]
[FSA-Red Algorithm]
[International Data Encryption Algorithm]
[Algorithmics]
[Common Scrambling Algorithm]
[Knuth's Algorithm X]
[Algorithmic probability]
[Algorithmic efficiency]
[Algorithmic inference]
[Lamport's Distributed Mutual Exclusion Algorithm]
[Algorithmic program debugging]
[Algorithmic learning theory]
[Algorithmic State Machine]
[Elliptic Curve Digital Signature Algorithm]
[Algorithmic Lovász local lemma]
[Generic Security Service Algorithm for Secret Key Transaction]
[Algorithmic information theory]
[Algorithm characterizations]
[Digital Signature Algorithm]
[Algorithm]
[Algorithmic mechanism design]
[GSP Algorithm]
[Algorithmic trading]
[Generalized Hebbian Algorithm]
[Shortest Path Faster Algorithm]
[Weighted Majority Algorithm]
[Data Authentication Algorithm]
[HMAC-based One-time Password Algorithm]
[Shinnar-Le Roux Algorithm]
[Algorithmic game theory]
[Algorithm engineering]
[Algorithmic skeleton]
[Secure Hash Algorithm]
[The Harmful Effects of Algorithms in Grades 1–4]
[Algorithmically random sequence]

 
 
 Field of Study "Computer" :
 
 [Torpedo Data Computer]
[Computer display standard]
[Computer programming]
[Computer-assisted personal interviewing]
[Computer memory]
[Network Computer]
[Information and Computer Science]
[Computer facial animation]
[Computer fan control]
[Computer literacy]
[Computer chess]
[Computer access control]
[Computer ethics]
[Computer forensics]
[Computerized maintenance management system]
[Computer data storage]
[Computer-mediated communication]
[Computer Engineering]
[Computer audition]
[Intergalactic Computer Network]
[Computer-generated imagery]
[Computer-supported cooperative work]
[Computer bridge]
[Computer-aided software engineering]
[Computer Modern]
[Computer representation of surfaces]
[Computer hardware]
[Computer network]
[Computer security compromised by hardware failure]
[Computer virus]
[AP Computer Science]
[Computer cluster]
[Computer graphics lighting]
[Computer-on-module]
[Computer module]
[Computer architecture]
[Computer fraud]
[Computer stereo vision]
[Computerized adaptive testing]
[Computer user satisfaction]
[Computer experiment]
[Computer-automated design]
[Computer fan]
[Computer file]
[Computer Graphics Metafile]
[Computerized classification test]
[Computer Automated Measurement and Control]
[Computer performance]
[Computer music]
[Computer network programming]
[Computer Aided Design]
[Computer number format]
[Computer security model]
[Computer Animation]
[Computer Science]
[Computer-assisted web interviewing]
[Computer optimization]
[Computer worm]
[Computer graphics]
[On the Cruelty of Really Teaching Computer Science]
[Computer-assisted proof]
[Computer-aided manufacturing]
[Computer-integrated manufacturing]
[Single-chip Cloud Computer]
[Computer cooling]
[Computer-assisted translation]
[Mark I Fire Control Computer]
[Computer Systems Research Group]
[Computer multitasking]
[Computer vision]
[Computer graphics (images)]
[Computer architecture simulator]
[Computer security]
[Computer-aided technologies]
[Computer simulation]
[Computerized system validation]
[Computer network operations]
[Computer for operations with functions]
[Computer port]
[Computer Applications]
[Integrated Computer-Aided Manufacturing]
[Computer-mediated reality]
[Computer terminal]
[Computer-aided engineering]
[Computer art]
[Computer appliance]
[Central Air Data Computer]

 
 */
