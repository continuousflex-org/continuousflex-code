=====================
ContinuousFlex plugin
=====================

This plugin provides HEMNMA and StructMap protocols and is frequently updated.


Installation
------------

You will need to use `2.0 <https://github.com/I2PC/scipion/releases/tag/V2.0.0>`_ version of Scipion to be able to run these protocols. To install the plugin, you have two options:

a) Stable version

   .. code-block::

      scipion installp -p scipion-em-continuousflex

b) Developer's version

   * download repository

   .. code-block::

      git clone https://github.com/scipion-em/scipion-em-continuousflex.git

   * install

   .. code-block::

      scipion installp -p path_to_scipion-em-continuousflex --devel

continuousflex sources will be downloaded automatically with the plugin,
but you can also link an existing installation.


Note: Xmipp plugin should be installed to run continuousflex. 

Supported versions
------------------

0.4

Protocols
---------

* HEMNMA: Hybrid Electron-Microscopy Normal-Mode-Analysis
* StructMap: Structural Mapping

References
----------

1. Jin Q, Sorzano CO, de la Rosa-Trevin JM, Bilbao-Castro JR, Nunez-Ramirez R, Llorca O, Tama F,Jonic S: Iterative elastic 3D-to-2D alignment method using normal modes for studying structural dynamics of large macromolecular complexes. Structure 2014, 22:496-506.
2. Sorzano CO, de la Rosa-Trevin JM, Tama F, Jonic S: Hybrid Electron Microscopy Normal Mode Analysis graphical interface and protocol. J Struct Biol 2014, 188:134-141.
3. Sanchez Sorzano CO, Alvarez-Cabrera AL, Kazemi M, Carazo JM, Jonic S: StructMap: Elastic Distance Analysis of Electron Microscopy Maps for Studying Conformational Changes. Biophys J 2016, 110:1753-1765.



# scipion-em-continuousflex
