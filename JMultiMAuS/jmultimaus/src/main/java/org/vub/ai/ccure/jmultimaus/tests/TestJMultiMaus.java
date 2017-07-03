package org.vub.ai.ccure.jmultimaus.tests;

import java.io.File;
import java.util.Arrays;

import org.jpy.PyLib;
import org.jpy.PyModule;
import org.jpy.PyObject;

/**
 * A simple test class to see if we can get the jpy bridge to work correctly and call some MultiMAuS python code
 * 
 * @author Dennis Soemers
 */
public class TestJMultiMaus {
	
	private static final String MULTIMAUS_PYTHON_FILEPATH = "..\\..";
	
	public static void main(String[] args){
		// TODO probably a good idea to actually put the entire jpy project inside our project
		// then we can modify the setup file a bit so that it always puts this file in a certain relative location,
		// and use that relative location as a filepath here instead of hardcoding the absolute filepath
		// can then also immediately include changes to the jpy pom.xml file for source and javadoc packaging
		System.setProperty("jpy.config", "D:\\Apps\\jpy\\build\\lib.win-amd64-3.6\\jpyconfig.properties");
		
		String multimausPythonAbsoluteFilepath = new File(MULTIMAUS_PYTHON_FILEPATH).getAbsolutePath();
		
		PyLib.startPython(multimausPythonAbsoluteFilepath);
		
		PyModule onlineSimulatorModule = PyModule.importModule("experiments.run_online_unimaus");
		PyObject simulator = onlineSimulatorModule.call("OnlineUnimaus");
		
		// generate data to prepare feature constructors
		simulator.callMethod("step_simulator");
		PyObject data = simulator.callMethod("get_log");
		simulator.callMethod("prepare_feature_constructors", data);
		
		// generate data that we can compute new features for
		simulator.callMethod("step_simulator");
		data = simulator.callMethod("get_log");
		simulator.callMethod("update_feature_constructors_unlabeled", data);
		
		// process the data (add features)
		data = simulator.callMethod("process_data", data);
		
		// wrap the data in an object with all the methods we need
		PyObject wrappedData = onlineSimulatorModule.call("DataLogWrapper", data);
		
		System.out.println("Column names = " + Arrays.toString(
				wrappedData.callMethod("get_column_names").
				getObjectArrayValue(String.class)));
		
		System.out.println("num rows = " + wrappedData.callMethod("get_num_rows").getIntValue());
		System.out.println("num cols = " + wrappedData.callMethod("get_num_cols").getIntValue());
		
		System.out.println();
		System.out.println("Data list = " + Arrays.toString(
				wrappedData.callMethod("get_data_list").
				getObjectArrayValue(Double.class)));
		
		PyLib.stopPython();
	}

}
