package org.vub.ai.ccure.jcredsim.tests;

import java.io.File;

import org.jpy.PyLib;
import org.jpy.PyModule;

/**
 * A simple test class to see if we can get the jpy bridge to work correctly and call some CredSim python code
 * 
 * @author Dennis Soemers
 */
public class TestJCredSim {
	
	private static final String CREDSIM_PYTHON_FILEPATH = "..\\..";
	
	public static void main(String[] args){
		// TODO probably a good idea to actually put the entire jpy project inside our project
		// then we can modify the setup file a bit so that it always puts this file in a certain relative location,
		// and use that relative location as a filepath here instead of hardcoding the absolute filepath
		// can then also immediately include changes to the jpy pom.xml file for source and javadoc packaging
		System.setProperty("jpy.config", "D:\\Apps\\jpy\\build\\lib.win-amd64-3.6\\jpyconfig.properties");
		
		String credsimPythonAbsoluteFilepath = new File(CREDSIM_PYTHON_FILEPATH).getAbsolutePath();
		
		PyLib.startPython(credsimPythonAbsoluteFilepath);
		
		PyModule simulatorModule = PyModule.importModule("simulator");
		simulatorModule.call("run_single()");
		
		PyLib.stopPython();
	}

}
