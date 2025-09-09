To use the action fit to calculate $\beta$ within *CosmoTransitions*, one can replace the *tunnelFromPhase* function in 
*transitionFinder.py*.

After that, the functionality can be invoked using the following code:
::
  m = model1()
  m.findAllTransitions()
    for tran in m.TnTrans:
      print('Tnuc=',tran['Tnuc'])
      print('action=',tran['action'])
      if 'd(S/T)/dT' in tran:
        print('d(S/T)/dT=',tran['d(S/T)/dT'])
