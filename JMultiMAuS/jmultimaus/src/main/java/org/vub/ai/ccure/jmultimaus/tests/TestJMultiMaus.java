package org.vub.ai.ccure.jmultimaus.tests;

import java.io.File;

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
		simulator.callMethod("step_simulator");
		System.out.println(simulator.callMethod("get_log").getAttribute("Global_Date"));
		
		PyLib.stopPython();
	}

}
