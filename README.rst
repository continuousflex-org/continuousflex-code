=====================
ContinuousFlex plugin
=====================

This plugin provides the latest Scipion protocols for cryo-EM continuous conformational flexibility/heterogeneity analysis of biomolecular complexes.


Installation
------------

You will need to use `3.0 <https://github.com/I2PC/scipion/releases>`_ version of Scipion to be able to run these protocols. To install the plugin, you have two options:
We you need help installing Scipion3, please refer to the Scipion Documentation `here <https://scipion-em.github.io/docs/docs/scipion-modes/how-to-install.html>`_

Make sure that you have cmake installed on your Linux system. For example, if you are using Ubuntu
 .. code-block::

    sudo apt install cmake


a) Stable version

	Install Scipion3 and use the plugin manager to install the plugin.

b) Developer's version

   * download repository

   .. code-block::

      git clone https://github.com/scipion-em/scipion-em-continuousflex.git
      git checkout devel

   * install

   .. code-block::

      scipion3 installp -p path_to_scipion-em-continuousflex --devel

continuousflex sources will be downloaded automatically with the plugin.


Note: Xmipp and Chimerax plugins should be installed (from Scipion3 plugin manager) to run continuousflex protocols.
You should also consider having VMD on your system for visualization.
We assume that VMD is installed on your system in "/usr/local/lib/vmd".
If VMD is installed but does not work, you may run the command "scipion3 config" and look for VMD_HOME in the config file (the config file is usually at ~/scipion3/config/scipion.conf)

Supported versions
------------------

versions > 3.0.15

Protocols
---------

* HEMNMA: Hybrid Electron-Microscopy Normal-Mode-Analysis
* HEMNMA-3D: Extension of HEMNMA for cryo-ET macromolecular continuous conformational variability analysis
* StructMap: Structural Mapping

Note: A reproduction of some utility codes with their corresponding licenses are contained in this plugin for subtomogram averaging, missing wedge correction, denoising and data reading. These codes are not used in the methods above, but they are made optional for data preprocessing and visualization.

References
----------
1. Harastani M, Sorzano CO, Jonić S. Hybrid Electron Microscopy Normal Mode Analysis with Scipion. Protein Science. 2020 Jan;29(1):223-36.
2. Harastani M, Eltsove M, Leforestier A, Jonić S. HEMNMA-3D: Cryo Electron Tomography Method Based on Normal Mode Analysis to Study Continuous Conformational Variability of Macromolecular Complexes. Front. Mol. Biosci 2021.
3. Jin Q, Sorzano CO, de la Rosa-Trevin JM, Bilbao-Castro JR, Nunez-Ramirez R, Llorca O, Tama F,Jonic S: Iterative elastic 3D-to-2D alignment method using normal modes for studying structural dynamics of large macromolecular complexes. Structure 2014, 22:496-506.
4. Sorzano CO, de la Rosa-Trevin JM, Tama F, Jonic S: Hybrid Electron Microscopy Normal Mode Analysis graphical interface and protocol. J Struct Biol 2014, 188:134-141.
5. Sanchez Sorzano CO, Alvarez-Cabrera AL, Kazemi M, Carazo JM, Jonic S: StructMap: Elastic Distance Analysis of Electron Microscopy Maps for Studying Conformational Changes. Biophys J 2016, 110:1753-1765.



# scipion-em-continuousflex
