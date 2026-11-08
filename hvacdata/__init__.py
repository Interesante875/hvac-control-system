"""Data preparation for the Smart Buildings HVAC control challenge.

Everything here wraps the official starter kit (``external/starter-kit``), which
holds the simulator, the building generator and the baselines. The kit resolves
its generated file paths relative to its own root, so every tool in this
package runs simulator code with the working directory set to the kit root;
use :func:`hvacdata.paths.in_kit` rather than ``os.chdir`` by hand.
"""
