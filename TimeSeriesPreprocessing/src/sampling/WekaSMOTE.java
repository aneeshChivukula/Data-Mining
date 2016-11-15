package sampling;

import java.io.File;
import java.io.IOException;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

public class WekaSMOTE {
	
	static String InDir = "/home/aneesh/Desktop/UTS\\ Literature\\ Survey/NASDAQ\\ Project/";
	static String csvDataFile = "threemonth_sample_demomarket_1000seriesalerts.csv";
	static String CSVFilePath = InDir + csvDataFile;
	static String OutFilePath = InDir + "temp.arff";
	
	static String converterOptionsCLI = "";
	
	public static void main(String[] args) throws Exception {		
		System.out.println(CSVFilePath);
		
		try {
			CSVLoader loader = new CSVLoader();
			loader.setSource(new File("/home/aneesh/Desktop/UTS Literature Survey/NASDAQ Project/threemonth_sample_demomarket_1000seriesalerts.csv"));
//			loader.setOptions(weka.core.Utils.splitOptions(converterOptionsCLI));
			Instances data = loader.getDataSet();
			System.out.println(data);

			
			ArffSaver saver = new ArffSaver();
			saver.setInstances(data);
			saver.setFile(new File(OutFilePath));
			saver.writeBatch();
			
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}

}
