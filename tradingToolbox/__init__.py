import pip
moduleList = [i.key for i in pip.get_installed_distributions()]

if 'quantiacstoolbox' in moduleList:
    from .quantiacsToolbox import   runTradingSystem, loadData, plotts, stats, submit, computeFees, updateCheck

__version__ = '2.2.0'
