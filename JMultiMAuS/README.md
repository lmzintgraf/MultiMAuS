# Installation Instructions jpy
---

jpy is the framework I'm going to try to use for making Java and Python communicate with each other. In this readme, I briefly 
describe installation instructions. May write them more cleanly / in more detail at a later point in time. 

I personally have Python 3.6, Java >= 8 and Windows 10 installed, so that is the only combination I can write instructions for. 
For other combinations, it may be useful to also have a look at [the official installation instructions of 
[jpy](http://jpy.readthedocs.io/en/latest/install.html). My instructions are mostly based on that, but I did need to take some 
other steps (primarily, I suspect, due to using Python 3.6 instead of 3.3).

## Requirements
---

- Python 3.3 or higher (possibly 3.5 or 3.6 or higher?)
- JDK 8 or higher (7 or higher or 6 or higher may also be sufficient, did not test)
- Maven 3 or higher
- Microsoft Visual Studio 2015 (I believe this is necessary from Python 3.5 onwards. For lower versions, Microsoft Visual 
Studio 2010 may be necessary instead)

## Instructions
---

1. Git Clone (or simply download) the [jpy project](https://github.com/bcdev/jpy).
2. Launch the Developer Command Prompt for VS2015 (think this is installed with VS2015 by default, but may be optional, not sure)
3. Navigate to <VS 2015 installation directory>\VC. By default, I believe <VS 2015 installation directory> should be something 
like "C:\Program Files\Microsoft Visual Studio 14.0\VC". In my case, due to changing the installation directory, it was
"D:\Apps\Microsoft Visual Studio 14.0". Note the "14.0" for VS2015, that is not a mistake.
4. Execute:
	> vcvarsall.bat x64
	
This step (and step 5) may actually not be necessary, but shouldn't hurt either. In my case it was necessary because I had
previously been messing around with VS2010 according to official installation instructions (based on older Python versions).
This step makes sure the compiler of VS2015 will be used, instead of the one of VS2010 (or any other).

5. Navigate to where you downloaded or cloned the jpy project in step 1.
6. Execute:
	> SET DISTUTILS_USE_SDK=1
	
	> SET JDK_HOME=<path to JDK installation directory, which should, for example, contain an "include" directory>
	
	> SET JAVA_HOME=%JDK_HOME%
	
	> SET PATH=%JDK_HOME%\jre\bin\server;%PATH%
	
7. Execute:
	> python setup.py --maven build
	
Assuming that the PATH is set up in such a way that it can find python (and possibly also maven, not sure), this should start
correctly building everything. This means compiling the java code and building the jars, and building some C++ code necessary
for the bridge. This process should create the following files for us (in addition to some cool .jars somewhere):

    build/
        lib-os-platform-python-version/
            jpy.so (Unixes only) jdl.so (Unixes only) jpy.pyd (Windows only) jdl.pyd (Windows only) jpyutil.py jpyconfig.py jpyconfig.properties	
			
It will also run some tests. It may also actually be useful to, instead, run:

	> python setup.py --maven build install
	
In addition to just building the files, it will also install the list of files mentioned above into 
<Python install directory\Lib\site-packages>, and install the .jar files and pom.xml into your local repository (so that a
different maven project with jpy as dependency will be able to find them). This seems useful to me, but did not find it anywhere
in the official instructions.

## Including Source and Javadoc in jpy Maven Package
---

Following the instructions above causes Maven to only build a .jar of the .class files for the Java part of the framework.
This is technically sufficient for using the framework, but annoying due to being unable to view the source code or view
Javadoc documentation within an IDE such as Eclipse. This can be fixed by editing the pom.xml file included in jpy prior
to running the "python setup.py --maven build install" command.

First, to include source files, within the <build><plugins></plugins></build> tags, add:

			<plugin>
				<groupId>org.apache.maven.plugins</groupId>
				<artifactId>maven-source-plugin</artifactId>
				<executions>
					<execution>
						<id>attach-sources</id>
						<goals>
							<goal>jar</goal>
						</goals>
					</execution>
				</executions>
			</plugin>
			
Next, to include javadoc, within the already existing <build><plugins><plugin></plugin></plugins></build> tags for 
maven-javadoc-plugin, add:

				<executions>
					<execution>
						<id>attach-javadocs</id>
						<goals>
							<goal>jar</goal>
						</goals>
					</execution>
				</executions>
				
It may also be necessary to add <groupId>org.apache.maven.plugins</groupId> right above the artifactId, not sure on this. Now,
after letting maven build and install, projects with jpy as maven dependency should be able to find javadoc and source code.