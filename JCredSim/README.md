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
- [Microsoft Windows SDK 7.1](https://www.microsoft.com/en-us/download/details.aspx?id=8279) or higher. Not sure actually
if this is really still necessary for Python >= 3.5, VS2015... but it is there in the official instructions.

## Instructions
---

1. Git Clone (or simply download) the [jpy project](https://github.com/bcdev/jpy).
2. Launch the Developer Command Prompt for VS2015 (think this is installed with VS2015 by default, but may be optional, not sure)
3. Navigate to <Microsoft Windows SDK 7.1 install directory>\Bin. By default, I believe <Microsoft Windows SDK 7.1 install 
directory> should be something like "C:\Program Files\Microsoft SDKs\Windows\v7.1".
4. Execute:
	> setenv /x64 /release
This is assuming we're on a 64-bit platform, and our installed JDK and Python versions are also 64-bit. If any of these are
not 64-bit, use /x86 instead of /x64. This is the step for which we needed to install that Microsoft Windows SDK 7.1. Just
because I don't know of another way to get access to that setenv command. I do suspect there should be an easier way to get it,
or simply do whatever it does... but this works for sure.
5. Navigate to <VS 2015 installation directory>\VC. By default, I believe <VS 2015 installation directory> should be something 
like "C:\Program Files\Microsoft Visual Studio 14.0\VC". In my case, due to changing the installation directory, it was
"D:\Apps\Microsoft Visual Studio 14.0". Note the "14.0" for VS2015, that is not a mistake.
6. Execute:
	> vcvarsall.bat
This step (and step 5) may actually not be necessary, but shouldn't hurt either. In my case it was necessary because I had
previously been messing around with VS2010 according to official installation instructions (based on older Python versions).
This step makes sure the compiler of VS2015 will be used, instead of the one of VS2010 (or any other).
7. Navigate to where you downloaded or cloned the jpy project in step 1.
8. Execute:
	> SET DISTUTILS_USE_SDK=1
	> SET JDK_HOME=<path to JDK installation directory, which should, for example, contain an "include" directory>
	> SET JAVA_HOME=%JDK_HOME%
	> SET PATH=%JDK_HOME%\jre\bin\server;%PATH%
9. Execute:
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